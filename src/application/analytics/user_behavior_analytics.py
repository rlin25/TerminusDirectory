"""
User Behavior Analytics for tracking and analyzing user interactions and patterns.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass
from enum import Enum
import asyncio
import pandas as pd
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import text
import redis.asyncio as aioredis


class UserSegment(Enum):
    """User segments for behavior analysis."""
    NEW_USER = "new_user"
    RETURNING_USER = "returning_user"
    POWER_USER = "power_user"
    CHURNED_USER = "churned_user"
    VIP_USER = "vip_user"


class InteractionType(Enum):
    """Types of user interactions."""
    SEARCH = "search"
    VIEW_PROPERTY = "view_property"
    SAVE_PROPERTY = "save_property"
    CONTACT_AGENT = "contact_agent"
    APPLY = "apply"
    SHARE = "share"
    FILTER = "filter"
    SORT = "sort"


@dataclass
class UserJourney:
    """Represents a user's journey through the platform."""
    user_id: str
    session_id: str
    start_time: datetime
    end_time: datetime
    interactions: List[Dict[str, Any]]
    conversion_events: List[Dict[str, Any]]
    devices: List[str]
    referrer: Optional[str] = None
    utm_source: Optional[str] = None


@dataclass
class BehaviorPattern:
    """Represents a discovered behavior pattern."""
    pattern_id: str
    pattern_name: str
    description: str
    frequency: int
    user_segments: List[UserSegment]
    interaction_sequence: List[InteractionType]
    conversion_rate: float
    avg_session_duration: float


@dataclass
class UserSegmentProfile:
    """Profile characteristics of a user segment."""
    segment: UserSegment
    user_count: int
    avg_session_duration: float
    avg_sessions_per_week: float
    avg_properties_viewed: float
    conversion_rate: float
    churn_rate: float
    preferred_property_types: List[str]
    preferred_locations: List[str]
    peak_activity_hours: List[int]


class UserBehaviorAnalytics:
    """
    Analyzes user behavior patterns, segments users, and tracks engagement metrics.
    
    Provides insights into user journeys, conversion funnels, and behavioral trends.
    """
    
    def __init__(
        self,
        db_session: AsyncSession,
        redis_client: aioredis.Redis,
        cache_ttl: int = 300  # 5 minutes
    ):
        self.db_session = db_session
        self.redis_client = redis_client
        self.cache_ttl = cache_ttl
    
    async def get_behavior_insights(
        self,
        time_range: str = "30d",
        segment: Optional[UserSegment] = None
    ) -> Dict[str, Any]:
        """Get comprehensive user behavior insights."""
        cache_key = f"behavior_insights:{time_range}:{segment}"
        
        # Try to get from cache
        cached_result = await self.redis_client.get(cache_key)
        if cached_result:
            return eval(cached_result)
        
        # Calculate time bounds
        end_time = datetime.utcnow()
        start_time = self._get_start_time(time_range, end_time)
        
        # Gather insights concurrently
        tasks = [
            self._get_user_engagement_metrics(start_time, end_time, segment),
            self._get_conversion_funnel_data(start_time, end_time, segment),
            self._get_popular_features(start_time, end_time, segment),
            self._get_session_analytics(start_time, end_time, segment),
            self._get_device_analytics(start_time, end_time, segment),
            self._get_geographic_distribution(start_time, end_time, segment)
        ]
        
        (engagement_metrics, funnel_data, popular_features, 
         session_analytics, device_analytics, geo_distribution) = await asyncio.gather(*tasks)
        
        insights = {
            "time_range": time_range,
            "segment": segment.value if segment else "all",
            "engagement_metrics": engagement_metrics,
            "conversion_funnel": funnel_data,
            "popular_features": popular_features,
            "session_analytics": session_analytics,
            "device_analytics": device_analytics,
            "geographic_distribution": geo_distribution,
            "generated_at": datetime.utcnow().isoformat()
        }
        
        # Cache the result
        await self.redis_client.setex(cache_key, self.cache_ttl, str(insights))
        
        return insights
    
    async def segment_users(self) -> Dict[UserSegment, UserSegmentProfile]:
        """Segment users based on behavior patterns and create profiles."""
        cache_key = "user_segments"
        
        # Try to get from cache
        cached_result = await self.redis_client.get(cache_key)
        if cached_result:
            return {UserSegment(k): UserSegmentProfile(**v) for k, v in eval(cached_result).items()}
        
        segments = {}
        
        # Define segmentation queries
        segmentation_queries = {
            UserSegment.NEW_USER: """
                SELECT user_id, COUNT(*) as session_count,
                       AVG(session_duration_minutes) as avg_duration,
                       MAX(created_at) as last_activity
                FROM user_sessions
                WHERE created_at > NOW() - INTERVAL '30 days'
                GROUP BY user_id
                HAVING COUNT(*) <= 3
                AND MAX(created_at) > NOW() - INTERVAL '7 days'
            """,
            
            UserSegment.RETURNING_USER: """
                SELECT user_id, COUNT(*) as session_count,
                       AVG(session_duration_minutes) as avg_duration,
                       MAX(created_at) as last_activity
                FROM user_sessions
                WHERE created_at > NOW() - INTERVAL '30 days'
                GROUP BY user_id
                HAVING COUNT(*) BETWEEN 4 AND 20
                AND MAX(created_at) > NOW() - INTERVAL '7 days'
            """,
            
            UserSegment.POWER_USER: """
                SELECT user_id, COUNT(*) as session_count,
                       AVG(session_duration_minutes) as avg_duration,
                       MAX(created_at) as last_activity
                FROM user_sessions
                WHERE created_at > NOW() - INTERVAL '30 days'
                GROUP BY user_id
                HAVING COUNT(*) > 20
                AND MAX(created_at) > NOW() - INTERVAL '3 days'
            """,
            
            UserSegment.CHURNED_USER: """
                SELECT user_id, COUNT(*) as session_count,
                       AVG(session_duration_minutes) as avg_duration,
                       MAX(created_at) as last_activity
                FROM user_sessions
                WHERE created_at > NOW() - INTERVAL '60 days'
                GROUP BY user_id
                HAVING MAX(created_at) < NOW() - INTERVAL '14 days'
                AND COUNT(*) > 5
            """
        }
        
        # Process each segment
        for segment_type, query in segmentation_queries.items():
            try:
                result = await self.db_session.execute(text(query))
                users = result.fetchall()
                
                if users:
                    # Calculate segment profile
                    profile = await self._calculate_segment_profile(segment_type, users)
                    segments[segment_type] = profile
                
            except Exception as e:
                print(f"Error processing segment {segment_type}: {e}")
        
        # Cache the result
        cache_data = {seg.value: profile.__dict__ for seg, profile in segments.items()}
        await self.redis_client.setex(cache_key, 3600, str(cache_data))  # 1 hour cache
        
        return segments
    
    async def analyze_user_journeys(
        self,
        user_ids: Optional[List[str]] = None,
        time_range: str = "7d"
    ) -> List[UserJourney]:
        """Analyze user journeys through the platform."""
        end_time = datetime.utcnow()
        start_time = self._get_start_time(time_range, end_time)
        
        # Build user filter
        user_filter = ""
        if user_ids:
            user_filter = f"AND user_id IN ({','.join(['%s'] * len(user_ids))})"
        
        # Get user sessions
        query = text(f"""
            SELECT 
                user_id,
                session_id,
                MIN(created_at) as start_time,
                MAX(created_at) as end_time,
                device_type,
                referrer,
                utm_source
            FROM user_sessions
            WHERE created_at BETWEEN :start_time AND :end_time
            {user_filter}
            GROUP BY user_id, session_id, device_type, referrer, utm_source
            ORDER BY user_id, start_time
        """)
        
        params = {"start_time": start_time, "end_time": end_time}
        if user_ids:
            params.update({f"user_{i}": uid for i, uid in enumerate(user_ids)})
        
        result = await self.db_session.execute(query, params)
        sessions = result.fetchall()
        
        journeys = []
        for session in sessions:
            # Get interactions for this session
            interactions = await self._get_session_interactions(session.session_id)
            conversion_events = await self._get_session_conversions(session.session_id)
            
            journey = UserJourney(
                user_id=session.user_id,
                session_id=session.session_id,
                start_time=session.start_time,
                end_time=session.end_time,
                interactions=interactions,
                conversion_events=conversion_events,
                devices=[session.device_type],
                referrer=session.referrer,
                utm_source=session.utm_source
            )
            journeys.append(journey)
        
        return journeys
    
    async def discover_behavior_patterns(
        self,
        min_frequency: int = 10,
        time_range: str = "30d"
    ) -> List[BehaviorPattern]:
        """Discover common behavior patterns in user interactions."""
        cache_key = f"behavior_patterns:{min_frequency}:{time_range}"
        
        # Try to get from cache
        cached_result = await self.redis_client.get(cache_key)
        if cached_result:
            return [BehaviorPattern(**p) for p in eval(cached_result)]
        
        end_time = datetime.utcnow()
        start_time = self._get_start_time(time_range, end_time)
        
        # Get interaction sequences
        query = text("""
            SELECT 
                session_id,
                user_id,
                interaction_type,
                created_at,
                properties_viewed,
                converted
            FROM user_interactions
            WHERE created_at BETWEEN :start_time AND :end_time
            ORDER BY session_id, created_at
        """)
        
        result = await self.db_session.execute(
            query,
            {"start_time": start_time, "end_time": end_time}
        )
        interactions = result.fetchall()
        
        # Group by session and create sequences
        session_sequences = {}
        for interaction in interactions:
            session_id = interaction.session_id
            if session_id not in session_sequences:
                session_sequences[session_id] = {
                    "user_id": interaction.user_id,
                    "sequence": [],
                    "converted": False
                }
            
            session_sequences[session_id]["sequence"].append(interaction.interaction_type)
            if interaction.converted:
                session_sequences[session_id]["converted"] = True
        
        # Find common patterns
        pattern_counts = {}
        conversion_counts = {}
        
        for session_data in session_sequences.values():
            sequence = tuple(session_data["sequence"])
            if len(sequence) >= 2:  # Minimum pattern length
                # Generate all subsequences of length 2-5
                for length in range(2, min(6, len(sequence) + 1)):
                    for i in range(len(sequence) - length + 1):
                        subseq = sequence[i:i+length]
                        pattern_counts[subseq] = pattern_counts.get(subseq, 0) + 1
                        
                        if session_data["converted"]:
                            conversion_counts[subseq] = conversion_counts.get(subseq, 0) + 1
        
        # Filter patterns by frequency and create BehaviorPattern objects
        patterns = []
        for pattern_seq, frequency in pattern_counts.items():
            if frequency >= min_frequency:
                conversion_rate = (conversion_counts.get(pattern_seq, 0) / frequency) * 100
                
                pattern = BehaviorPattern(
                    pattern_id=f"pattern_{hash(pattern_seq)}",
                    pattern_name=f"Pattern: {' -> '.join(pattern_seq)}",
                    description=f"Common sequence of {len(pattern_seq)} interactions",
                    frequency=frequency,
                    user_segments=[],  # Would be determined by analyzing users following this pattern
                    interaction_sequence=[InteractionType(step) for step in pattern_seq 
                                        if step in [t.value for t in InteractionType]],
                    conversion_rate=conversion_rate,
                    avg_session_duration=0.0  # Would be calculated from session data
                )
                patterns.append(pattern)
        
        # Sort by frequency
        patterns.sort(key=lambda p: p.frequency, reverse=True)
        
        # Cache the result
        cache_data = [p.__dict__ for p in patterns]
        await self.redis_client.setex(cache_key, 3600, str(cache_data))  # 1 hour cache
        
        return patterns[:50]  # Return top 50 patterns
    
    async def get_conversion_funnel(
        self,
        funnel_steps: List[str],
        time_range: str = "30d",
        segment: Optional[UserSegment] = None
    ) -> Dict[str, Any]:
        """Analyze conversion funnel for specified steps."""
        cache_key = f"conversion_funnel:{hash(str(funnel_steps))}:{time_range}:{segment}"
        
        # Try to get from cache
        cached_result = await self.redis_client.get(cache_key)
        if cached_result:
            return eval(cached_result)
        
        end_time = datetime.utcnow()
        start_time = self._get_start_time(time_range, end_time)
        
        # Build segment filter
        segment_filter = ""
        if segment:
            segment_users = await self._get_segment_users(segment)
            if segment_users:
                segment_filter = f"AND user_id IN ({','.join(['%s'] * len(segment_users))})"
        
        funnel_data = {
            "steps": funnel_steps,
            "conversions": [],
            "drop_off_rates": [],
            "total_users": 0
        }
        
        previous_count = None
        for i, step in enumerate(funnel_steps):
            # Count users who completed this step
            query = text(f"""
                SELECT COUNT(DISTINCT user_id) as user_count
                FROM user_interactions
                WHERE interaction_type = :step
                AND created_at BETWEEN :start_time AND :end_time
                {segment_filter}
            """)
            
            params = {"step": step, "start_time": start_time, "end_time": end_time}
            
            result = await self.db_session.execute(query, params)
            row = result.fetchone()
            current_count = row.user_count or 0
            
            if i == 0:
                funnel_data["total_users"] = current_count
                previous_count = current_count
            
            conversion_rate = (current_count / funnel_data["total_users"]) * 100 if funnel_data["total_users"] > 0 else 0
            drop_off_rate = ((previous_count - current_count) / previous_count) * 100 if previous_count > 0 else 0
            
            funnel_data["conversions"].append({
                "step": step,
                "users": current_count,
                "conversion_rate": conversion_rate,
                "step_conversion_rate": (current_count / previous_count) * 100 if previous_count > 0 else 100
            })
            
            if i > 0:
                funnel_data["drop_off_rates"].append({
                    "from_step": funnel_steps[i-1],
                    "to_step": step,
                    "drop_off_rate": drop_off_rate
                })
            
            previous_count = current_count
        
        # Cache the result
        await self.redis_client.setex(cache_key, self.cache_ttl, str(funnel_data))
        
        return funnel_data
    
    async def predict_user_churn(
        self,
        user_ids: Optional[List[str]] = None
    ) -> Dict[str, float]:
        """Predict churn probability for users based on behavior patterns."""
        # This is a simplified churn prediction
        # In production, you'd use a trained ML model
        
        churn_scores = {}
        
        # Get recent user activity
        query = text("""
            SELECT 
                user_id,
                COUNT(*) as session_count,
                MAX(created_at) as last_activity,
                AVG(session_duration_minutes) as avg_duration
            FROM user_sessions
            WHERE created_at > NOW() - INTERVAL '30 days'
            GROUP BY user_id
        """)
        
        result = await self.db_session.execute(query)
        users = result.fetchall()
        
        for user in users:
            if user_ids and user.user_id not in user_ids:
                continue
            
            # Simple heuristic-based churn score
            days_since_last_activity = (datetime.utcnow() - user.last_activity).days
            
            churn_score = 0.0
            
            # Factor 1: Days since last activity
            if days_since_last_activity > 14:
                churn_score += 0.4
            elif days_since_last_activity > 7:
                churn_score += 0.2
            
            # Factor 2: Session frequency
            if user.session_count < 3:
                churn_score += 0.3
            elif user.session_count < 10:
                churn_score += 0.1
            
            # Factor 3: Session duration
            if user.avg_duration < 2:  # Less than 2 minutes
                churn_score += 0.2
            elif user.avg_duration < 5:
                churn_score += 0.1
            
            churn_scores[user.user_id] = min(1.0, churn_score)
        
        return churn_scores
    
    # Private helper methods
    def _get_start_time(self, time_range: str, end_time: datetime) -> datetime:
        """Calculate start time based on time range."""
        if time_range == "1h":
            return end_time - timedelta(hours=1)
        elif time_range == "24h":
            return end_time - timedelta(days=1)
        elif time_range == "7d":
            return end_time - timedelta(days=7)
        elif time_range == "30d":
            return end_time - timedelta(days=30)
        elif time_range == "90d":
            return end_time - timedelta(days=90)
        else:
            return end_time - timedelta(days=30)  # Default to 30 days
    
    async def _get_user_engagement_metrics(
        self,
        start_time: datetime,
        end_time: datetime,
        segment: Optional[UserSegment]
    ) -> Dict[str, Any]:
        """Calculate user engagement metrics."""
        segment_filter = ""
        if segment:
            segment_users = await self._get_segment_users(segment)
            if segment_users:
                segment_filter = f"AND user_id IN ({','.join(['%s'] * len(segment_users))})"
        
        query = text(f"""
            SELECT 
                COUNT(DISTINCT user_id) as unique_users,
                COUNT(*) as total_sessions,
                AVG(session_duration_minutes) as avg_session_duration,
                SUM(page_views) as total_page_views,
                AVG(page_views) as avg_page_views_per_session,
                COUNT(CASE WHEN bounce = true THEN 1 END)::float / COUNT(*)::float as bounce_rate
            FROM user_sessions
            WHERE created_at BETWEEN :start_time AND :end_time
            {segment_filter}
        """)
        
        result = await self.db_session.execute(
            query,
            {"start_time": start_time, "end_time": end_time}
        )
        row = result.fetchone()
        
        return {
            "unique_users": row.unique_users or 0,
            "total_sessions": row.total_sessions or 0,
            "avg_session_duration": float(row.avg_session_duration or 0),
            "total_page_views": row.total_page_views or 0,
            "avg_page_views_per_session": float(row.avg_page_views_per_session or 0),
            "bounce_rate": float(row.bounce_rate or 0) * 100
        }
    
    async def _get_conversion_funnel_data(
        self,
        start_time: datetime,
        end_time: datetime,
        segment: Optional[UserSegment]
    ) -> Dict[str, Any]:
        """Get default conversion funnel data."""
        default_funnel = ["search", "view_property", "save_property", "contact_agent", "apply"]
        return await self.get_conversion_funnel(default_funnel, "30d", segment)
    
    async def _get_popular_features(
        self,
        start_time: datetime,
        end_time: datetime,
        segment: Optional[UserSegment]
    ) -> List[Dict[str, Any]]:
        """Get most popular features/interactions."""
        segment_filter = ""
        if segment:
            segment_users = await self._get_segment_users(segment)
            if segment_users:
                segment_filter = f"AND user_id IN ({','.join(['%s'] * len(segment_users))})"
        
        query = text(f"""
            SELECT 
                interaction_type,
                COUNT(*) as usage_count,
                COUNT(DISTINCT user_id) as unique_users
            FROM user_interactions
            WHERE created_at BETWEEN :start_time AND :end_time
            {segment_filter}
            GROUP BY interaction_type
            ORDER BY usage_count DESC
            LIMIT 10
        """)
        
        result = await self.db_session.execute(
            query,
            {"start_time": start_time, "end_time": end_time}
        )
        features = result.fetchall()
        
        return [
            {
                "feature": feature.interaction_type,
                "usage_count": feature.usage_count,
                "unique_users": feature.unique_users
            }
            for feature in features
        ]
    
    async def _get_session_analytics(
        self,
        start_time: datetime,
        end_time: datetime,
        segment: Optional[UserSegment]
    ) -> Dict[str, Any]:
        """Get session-level analytics."""
        # Implementation for session analytics
        return {
            "avg_sessions_per_user": 0.0,
            "session_duration_distribution": {},
            "peak_hours": []
        }
    
    async def _get_device_analytics(
        self,
        start_time: datetime,
        end_time: datetime,
        segment: Optional[UserSegment]
    ) -> Dict[str, Any]:
        """Get device and platform analytics."""
        # Implementation for device analytics
        return {
            "device_breakdown": {},
            "browser_breakdown": {},
            "os_breakdown": {}
        }
    
    async def _get_geographic_distribution(
        self,
        start_time: datetime,
        end_time: datetime,
        segment: Optional[UserSegment]
    ) -> Dict[str, Any]:
        """Get geographic distribution of users."""
        # Implementation for geographic analytics
        return {
            "countries": {},
            "cities": {},
            "regions": {}
        }
    
    async def _calculate_segment_profile(
        self,
        segment_type: UserSegment,
        users: List[Any]
    ) -> UserSegmentProfile:
        """Calculate profile for a user segment."""
        # This would involve more complex calculations
        # For now, return a basic profile
        return UserSegmentProfile(
            segment=segment_type,
            user_count=len(users),
            avg_session_duration=0.0,
            avg_sessions_per_week=0.0,
            avg_properties_viewed=0.0,
            conversion_rate=0.0,
            churn_rate=0.0,
            preferred_property_types=[],
            preferred_locations=[],
            peak_activity_hours=[]
        )
    
    async def _get_segment_users(self, segment: UserSegment) -> List[str]:
        """Get list of user IDs for a specific segment."""
        # This would implement the actual segment logic
        return []
    
    async def _get_session_interactions(self, session_id: str) -> List[Dict[str, Any]]:
        """Get all interactions for a session."""
        query = text("""
            SELECT interaction_type, created_at, metadata
            FROM user_interactions
            WHERE session_id = :session_id
            ORDER BY created_at
        """)
        
        result = await self.db_session.execute(query, {"session_id": session_id})
        interactions = result.fetchall()
        
        return [
            {
                "type": interaction.interaction_type,
                "timestamp": interaction.created_at,
                "metadata": interaction.metadata
            }
            for interaction in interactions
        ]
    
    async def _get_session_conversions(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversion events for a session."""
        query = text("""
            SELECT event_type, created_at, value
            FROM conversion_events
            WHERE session_id = :session_id
            ORDER BY created_at
        """)
        
        result = await self.db_session.execute(query, {"session_id": session_id})
        conversions = result.fetchall()
        
        return [
            {
                "type": conversion.event_type,
                "timestamp": conversion.created_at,
                "value": conversion.value
            }
            for conversion in conversions
        ]