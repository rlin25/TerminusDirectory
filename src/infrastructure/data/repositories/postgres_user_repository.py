import asyncio
import logging
from typing import List, Optional, Dict, Any
from uuid import UUID
from datetime import datetime
import asyncpg
from asyncpg import Pool

from ....domain.entities.user import User, UserPreferences, UserInteraction
from ....domain.repositories.user_repository import UserRepository


class PostgreSQLUserRepository(UserRepository):
    """PostgreSQL implementation of UserRepository with async operations"""
    
    def __init__(self, connection_pool: Pool):
        self.pool = connection_pool
        self.logger = logging.getLogger(__name__)
    
    async def get_by_id(self, user_id: UUID) -> Optional[User]:
        """Get user by ID"""
        try:
            async with self.pool.acquire() as conn:
                # Get user basic info
                user_row = await conn.fetchrow(
                    """
                    SELECT id, email, created_at, is_active,
                           min_price, max_price, min_bedrooms, max_bedrooms,
                           min_bathrooms, max_bathrooms, preferred_locations,
                           required_amenities, property_types
                    FROM users 
                    WHERE id = $1
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
                    is_active=user_row['is_active']
                )
                
                return user
                
        except Exception as e:
            self.logger.error(f"Failed to get user {user_id}: {e}")
            return None
    
    async def get_by_email(self, email: str) -> Optional[User]:
        """Get user by email address"""
        try:
            async with self.pool.acquire() as conn:
                user_row = await conn.fetchrow(
                    "SELECT id FROM users WHERE email = $1",
                    email
                )
                
                if user_row:
                    return await self.get_by_id(user_row['id'])
                return None
                
        except Exception as e:
            self.logger.error(f"Failed to get user by email {email}: {e}")
            return None
    
    async def save(self, user: User) -> bool:
        """Save or update user"""
        try:
            async with self.pool.acquire() as conn:
                async with conn.transaction():
                    # Check if user exists
                    exists = await conn.fetchval(
                        "SELECT EXISTS(SELECT 1 FROM users WHERE id = $1)",
                        user.id
                    )
                    
                    if exists:
                        # Update existing user
                        await conn.execute(
                            """
                            UPDATE users SET
                                email = $2,
                                is_active = $3,
                                min_price = $4,
                                max_price = $5,
                                min_bedrooms = $6,
                                max_bedrooms = $7,
                                min_bathrooms = $8,
                                max_bathrooms = $9,
                                preferred_locations = $10,
                                required_amenities = $11,
                                property_types = $12,
                                updated_at = CURRENT_TIMESTAMP
                            WHERE id = $1
                            """,
                            user.id,
                            user.email,
                            user.is_active,
                            user.preferences.min_price,
                            user.preferences.max_price,
                            user.preferences.min_bedrooms,
                            user.preferences.max_bedrooms,
                            user.preferences.min_bathrooms,
                            user.preferences.max_bathrooms,
                            user.preferences.preferred_locations,
                            user.preferences.required_amenities,
                            user.preferences.property_types
                        )
                    else:
                        # Insert new user
                        await conn.execute(
                            """
                            INSERT INTO users (
                                id, email, created_at, is_active,
                                min_price, max_price, min_bedrooms, max_bedrooms,
                                min_bathrooms, max_bathrooms, preferred_locations,
                                required_amenities, property_types
                            ) VALUES (
                                $1, $2, $3, $4, $5, $6, $7, $8, $9, $10, $11, $12, $13
                            )
                            """,
                            user.id,
                            user.email,
                            user.created_at,
                            user.is_active,
                            user.preferences.min_price,
                            user.preferences.max_price,
                            user.preferences.min_bedrooms,
                            user.preferences.max_bedrooms,
                            user.preferences.min_bathrooms,
                            user.preferences.max_bathrooms,
                            user.preferences.preferred_locations,
                            user.preferences.required_amenities,
                            user.preferences.property_types
                        )
                    
                    # Save interactions
                    await self._save_user_interactions(conn, user.id, user.interactions)
                    
                    return True
                    
        except Exception as e:
            self.logger.error(f"Failed to save user {user.id}: {e}")
            return False
    
    async def delete(self, user_id: UUID) -> bool:
        """Delete user (soft delete)"""
        try:
            async with self.pool.acquire() as conn:
                result = await conn.execute(
                    """
                    UPDATE users SET 
                        is_active = false,
                        updated_at = CURRENT_TIMESTAMP
                    WHERE id = $1
                    """,
                    user_id
                )
                
                return result == "UPDATE 1"
                
        except Exception as e:
            self.logger.error(f"Failed to delete user {user_id}: {e}")
            return False
    
    async def add_interaction(self, user_id: UUID, interaction: UserInteraction) -> bool:
        """Add user interaction"""
        try:
            async with self.pool.acquire() as conn:
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
                
                return True
                
        except Exception as e:
            self.logger.error(f"Failed to add interaction for user {user_id}: {e}")
            return False
    
    async def get_interactions(self, user_id: UUID, 
                             interaction_type: Optional[str] = None,
                             limit: int = 100) -> List[UserInteraction]:
        """Get user interactions with optional filtering"""
        try:
            async with self.pool.acquire() as conn:
                return await self._get_user_interactions(
                    conn, user_id, interaction_type, limit
                )
                
        except Exception as e:
            self.logger.error(f"Failed to get interactions for user {user_id}: {e}")
            return []
    
    async def get_similar_users(self, user_id: UUID, limit: int = 10) -> List[User]:
        """Find users with similar preferences and interactions"""
        try:
            async with self.pool.acquire() as conn:
                # Get target user preferences
                target_user = await self.get_by_id(user_id)
                if not target_user:
                    return []
                
                # Find users with similar preferences using similarity scoring
                similar_user_rows = await conn.fetch(
                    """
                    WITH target_prefs AS (
                        SELECT $1 as target_user_id,
                               $2::text[] as target_locations,
                               $3::text[] as target_amenities,
                               $4 as target_min_price,
                               $5 as target_max_price
                    ),
                    user_similarity AS (
                        SELECT 
                            u.id,
                            u.email,
                            -- Location similarity
                            CASE 
                                WHEN u.preferred_locations && tp.target_locations THEN 2
                                ELSE 0
                            END +
                            -- Amenity similarity  
                            CASE
                                WHEN u.required_amenities && tp.target_amenities THEN 2
                                ELSE 0
                            END +
                            -- Price range overlap
                            CASE
                                WHEN (u.min_price IS NULL OR tp.target_max_price IS NULL OR u.min_price <= tp.target_max_price)
                                 AND (u.max_price IS NULL OR tp.target_min_price IS NULL OR u.max_price >= tp.target_min_price)
                                THEN 1
                                ELSE 0
                            END as similarity_score
                        FROM users u
                        CROSS JOIN target_prefs tp
                        WHERE u.id != tp.target_user_id 
                          AND u.is_active = true
                    )
                    SELECT id, similarity_score
                    FROM user_similarity
                    WHERE similarity_score > 0
                    ORDER BY similarity_score DESC, id
                    LIMIT $6
                    """,
                    user_id,
                    target_user.preferences.preferred_locations,
                    target_user.preferences.required_amenities,
                    target_user.preferences.min_price,
                    target_user.preferences.max_price,
                    limit
                )
                
                # Fetch full user objects
                similar_users = []
                for row in similar_user_rows:
                    user = await self.get_by_id(row['id'])
                    if user:
                        similar_users.append(user)
                
                return similar_users
                
        except Exception as e:
            self.logger.error(f"Failed to get similar users for {user_id}: {e}")
            return []
    
    async def get_user_statistics(self, user_id: UUID) -> Dict[str, Any]:
        """Get user activity statistics"""
        try:
            async with self.pool.acquire() as conn:
                stats = await conn.fetchrow(
                    """
                    SELECT 
                        COUNT(*) as total_interactions,
                        COUNT(DISTINCT property_id) as unique_properties,
                        COUNT(*) FILTER (WHERE interaction_type = 'view') as views,
                        COUNT(*) FILTER (WHERE interaction_type = 'like') as likes,
                        COUNT(*) FILTER (WHERE interaction_type = 'inquiry') as inquiries,
                        MAX(timestamp) as last_activity,
                        MIN(timestamp) as first_activity
                    FROM user_interactions
                    WHERE user_id = $1
                    """,
                    user_id
                )
                
                return {
                    'total_interactions': stats['total_interactions'],
                    'unique_properties': stats['unique_properties'],
                    'views': stats['views'],
                    'likes': stats['likes'],
                    'inquiries': stats['inquiries'],
                    'last_activity': stats['last_activity'],
                    'first_activity': stats['first_activity']
                }
                
        except Exception as e:
            self.logger.error(f"Failed to get statistics for user {user_id}: {e}")
            return {}
    
    async def _get_user_interactions(self, conn, user_id: UUID, 
                                   interaction_type: Optional[str] = None,
                                   limit: int = 100) -> List[UserInteraction]:
        """Helper to get user interactions from database connection"""
        query = """
            SELECT property_id, interaction_type, timestamp, duration_seconds
            FROM user_interactions
            WHERE user_id = $1
        """
        params = [user_id]
        
        if interaction_type:
            query += " AND interaction_type = $2"
            params.append(interaction_type)
        
        query += " ORDER BY timestamp DESC LIMIT $" + str(len(params) + 1)
        params.append(limit)
        
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
        """Helper to save user interactions"""
        # Delete existing interactions for this user
        await conn.execute(
            "DELETE FROM user_interactions WHERE user_id = $1",
            user_id
        )
        
        # Insert new interactions
        if interactions:
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
                """,
                interaction_data
            )
    
    async def get_active_users_count(self) -> int:
        """Get count of active users"""
        try:
            async with self.pool.acquire() as conn:
                count = await conn.fetchval(
                    "SELECT COUNT(*) FROM users WHERE is_active = true"
                )
                return count
        except Exception as e:
            self.logger.error(f"Failed to get active users count: {e}")
            return 0
    
    async def get_users_by_preference_location(self, location: str, 
                                             limit: int = 50) -> List[User]:
        """Get users who prefer a specific location"""
        try:
            async with self.pool.acquire() as conn:
                user_ids = await conn.fetch(
                    """
                    SELECT id FROM users 
                    WHERE is_active = true 
                      AND $1 = ANY(preferred_locations)
                    LIMIT $2
                    """,
                    location,
                    limit
                )
                
                users = []
                for row in user_ids:
                    user = await self.get_by_id(row['id'])
                    if user:
                        users.append(user)
                
                return users
                
        except Exception as e:
            self.logger.error(f"Failed to get users by location {location}: {e}")
            return []