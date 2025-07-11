"""
Unit tests for the User domain entity and related classes.

Tests User, UserPreferences, and UserInteraction entities.
"""

import pytest
from datetime import datetime, timedelta
from uuid import UUID, uuid4
from typing import List

from domain.entities.user import User, UserPreferences, UserInteraction
from tests.utils.data_factories import UserFactory, PropertyFactory, FactoryConfig


class TestUserPreferences:
    """Test cases for UserPreferences entity."""
    
    def test_user_preferences_creation_with_defaults(self):
        """Test UserPreferences creation with default values."""
        preferences = UserPreferences()
        
        assert preferences.min_price is None
        assert preferences.max_price is None
        assert preferences.min_bedrooms is None
        assert preferences.max_bedrooms is None
        assert preferences.min_bathrooms is None
        assert preferences.max_bathrooms is None
        assert preferences.preferred_locations == []
        assert preferences.required_amenities == []
        assert preferences.property_types == ["apartment"]
    
    def test_user_preferences_creation_with_values(self):
        """Test UserPreferences creation with specific values."""
        preferences = UserPreferences(
            min_price=2000.0,
            max_price=5000.0,
            min_bedrooms=2,
            max_bedrooms=4,
            min_bathrooms=1.5,
            max_bathrooms=3.0,
            preferred_locations=["Downtown", "Mission"],
            required_amenities=["parking", "gym"],
            property_types=["apartment", "condo"]
        )
        
        assert preferences.min_price == 2000.0
        assert preferences.max_price == 5000.0
        assert preferences.min_bedrooms == 2
        assert preferences.max_bedrooms == 4
        assert preferences.min_bathrooms == 1.5
        assert preferences.max_bathrooms == 3.0
        assert preferences.preferred_locations == ["Downtown", "Mission"]
        assert preferences.required_amenities == ["parking", "gym"]
        assert preferences.property_types == ["apartment", "condo"]
    
    def test_user_preferences_post_init(self):
        """Test UserPreferences __post_init__ method."""
        # Test with None values - should be converted to empty lists/defaults
        preferences = UserPreferences(
            preferred_locations=None,
            required_amenities=None,
            property_types=None
        )
        
        assert preferences.preferred_locations == []
        assert preferences.required_amenities == []
        assert preferences.property_types == ["apartment"]
    
    def test_user_preferences_partial_specification(self):
        """Test UserPreferences with partial specification."""
        preferences = UserPreferences(
            min_price=1500.0,
            max_bedrooms=3,
            preferred_locations=["SoMa"]
        )
        
        assert preferences.min_price == 1500.0
        assert preferences.max_price is None
        assert preferences.min_bedrooms is None
        assert preferences.max_bedrooms == 3
        assert preferences.preferred_locations == ["SoMa"]
        assert preferences.required_amenities == []
        assert preferences.property_types == ["apartment"]


class TestUserInteraction:
    """Test cases for UserInteraction entity."""
    
    def test_user_interaction_creation(self):
        """Test UserInteraction creation."""
        property_id = uuid4()
        interaction = UserInteraction(
            property_id=property_id,
            interaction_type="view",
            timestamp=datetime.now(),
            duration_seconds=120
        )
        
        assert interaction.property_id == property_id
        assert interaction.interaction_type == "view"
        assert isinstance(interaction.timestamp, datetime)
        assert interaction.duration_seconds == 120
    
    def test_user_interaction_create_class_method(self):
        """Test UserInteraction.create() class method."""
        property_id = uuid4()
        interaction = UserInteraction.create(
            property_id=property_id,
            interaction_type="like",
            duration_seconds=60
        )
        
        assert interaction.property_id == property_id
        assert interaction.interaction_type == "like"
        assert interaction.duration_seconds == 60
        assert isinstance(interaction.timestamp, datetime)
        
        # Timestamp should be recent
        time_diff = datetime.now() - interaction.timestamp
        assert time_diff.total_seconds() < 5
    
    def test_user_interaction_create_without_duration(self):
        """Test UserInteraction.create() without duration."""
        property_id = uuid4()
        interaction = UserInteraction.create(
            property_id=property_id,
            interaction_type="save"
        )
        
        assert interaction.property_id == property_id
        assert interaction.interaction_type == "save"
        assert interaction.duration_seconds is None
        assert isinstance(interaction.timestamp, datetime)
    
    def test_interaction_types(self):
        """Test various interaction types."""
        property_id = uuid4()
        interaction_types = ["view", "like", "inquiry", "save", "dislike", "share"]
        
        for interaction_type in interaction_types:
            interaction = UserInteraction.create(
                property_id=property_id,
                interaction_type=interaction_type
            )
            assert interaction.interaction_type == interaction_type


class TestUser:
    """Test cases for User entity."""
    
    def setup_method(self):
        """Set up test fixtures for each test."""
        self.user_factory = UserFactory(FactoryConfig(seed=42))
        self.property_factory = PropertyFactory(FactoryConfig(seed=42))
    
    def test_user_creation_with_factory(self):
        """Test user creation using factory."""
        user = self.user_factory.create()
        
        assert isinstance(user, User)
        assert isinstance(user.id, UUID)
        assert user.email
        assert "@" in user.email
        assert isinstance(user.preferences, UserPreferences)
        assert isinstance(user.interactions, list)
        assert len(user.interactions) == 0  # No interactions initially
        assert isinstance(user.created_at, datetime)
        assert user.is_active is True
    
    def test_user_create_class_method(self):
        """Test User.create() class method."""
        preferences = UserPreferences(
            min_price=2000.0,
            max_price=4000.0,
            preferred_locations=["Downtown"]
        )
        
        user = User.create(
            email="test@example.com",
            preferences=preferences
        )
        
        assert user.email == "test@example.com"
        assert user.preferences == preferences
        assert isinstance(user.id, UUID)
        assert isinstance(user.created_at, datetime)
        assert user.is_active is True
        assert user.interactions == []
    
    def test_user_create_with_default_preferences(self):
        """Test User.create() with default preferences."""
        user = User.create(email="test@example.com")
        
        assert user.email == "test@example.com"
        assert isinstance(user.preferences, UserPreferences)
        assert user.preferences.property_types == ["apartment"]
        assert user.preferences.preferred_locations == []
    
    def test_add_interaction(self):
        """Test adding interactions to user."""
        user = self.user_factory.create()
        property_id = uuid4()
        
        # Add view interaction
        view_interaction = UserInteraction.create(
            property_id=property_id,
            interaction_type="view",
            duration_seconds=120
        )
        user.add_interaction(view_interaction)
        
        assert len(user.interactions) == 1
        assert user.interactions[0] == view_interaction
        
        # Add like interaction
        like_interaction = UserInteraction.create(
            property_id=property_id,
            interaction_type="like"
        )
        user.add_interaction(like_interaction)
        
        assert len(user.interactions) == 2
        assert user.interactions[1] == like_interaction
    
    def test_get_interaction_history_all(self):
        """Test getting all interaction history."""
        user = self.user_factory.create()
        properties = self.property_factory.create_batch(3)
        
        # Add various interactions
        interactions = [
            UserInteraction.create(properties[0].id, "view"),
            UserInteraction.create(properties[1].id, "like"),
            UserInteraction.create(properties[2].id, "save"),
            UserInteraction.create(properties[0].id, "inquiry")
        ]
        
        for interaction in interactions:
            user.add_interaction(interaction)
        
        history = user.get_interaction_history()
        assert len(history) == 4
        assert all(interaction in history for interaction in interactions)
    
    def test_get_interaction_history_by_type(self):
        """Test getting interaction history filtered by type."""
        user = self.user_factory.create()
        properties = self.property_factory.create_batch(3)
        
        # Add various interactions
        interactions = [
            UserInteraction.create(properties[0].id, "view"),
            UserInteraction.create(properties[1].id, "like"),
            UserInteraction.create(properties[2].id, "view"),
            UserInteraction.create(properties[0].id, "save")
        ]
        
        for interaction in interactions:
            user.add_interaction(interaction)
        
        # Test filtering by "view"
        view_history = user.get_interaction_history("view")
        assert len(view_history) == 2
        assert all(interaction.interaction_type == "view" for interaction in view_history)
        
        # Test filtering by "like"
        like_history = user.get_interaction_history("like")
        assert len(like_history) == 1
        assert like_history[0].interaction_type == "like"
        
        # Test filtering by non-existent type
        missing_history = user.get_interaction_history("nonexistent")
        assert len(missing_history) == 0
    
    def test_get_liked_properties(self):
        """Test getting liked properties."""
        user = self.user_factory.create()
        properties = self.property_factory.create_batch(4)
        
        # Add various interactions
        interactions = [
            UserInteraction.create(properties[0].id, "view"),
            UserInteraction.create(properties[1].id, "like"),
            UserInteraction.create(properties[2].id, "like"),
            UserInteraction.create(properties[3].id, "save")
        ]
        
        for interaction in interactions:
            user.add_interaction(interaction)
        
        liked_properties = user.get_liked_properties()
        assert len(liked_properties) == 2
        assert properties[1].id in liked_properties
        assert properties[2].id in liked_properties
        assert properties[0].id not in liked_properties
        assert properties[3].id not in liked_properties
    
    def test_get_viewed_properties(self):
        """Test getting viewed properties."""
        user = self.user_factory.create()
        properties = self.property_factory.create_batch(4)
        
        # Add various interactions
        interactions = [
            UserInteraction.create(properties[0].id, "view"),
            UserInteraction.create(properties[1].id, "like"),
            UserInteraction.create(properties[2].id, "view"),
            UserInteraction.create(properties[3].id, "save")
        ]
        
        for interaction in interactions:
            user.add_interaction(interaction)
        
        viewed_properties = user.get_viewed_properties()
        assert len(viewed_properties) == 2
        assert properties[0].id in viewed_properties
        assert properties[2].id in viewed_properties
        assert properties[1].id not in viewed_properties
        assert properties[3].id not in viewed_properties
    
    def test_update_preferences(self):
        """Test updating user preferences."""
        user = self.user_factory.create()
        original_preferences = user.preferences
        
        new_preferences = UserPreferences(
            min_price=3000.0,
            max_price=6000.0,
            preferred_locations=["Pacific Heights"],
            required_amenities=["parking", "pool"]
        )
        
        user.update_preferences(new_preferences)
        
        assert user.preferences == new_preferences
        assert user.preferences != original_preferences
        assert user.preferences.min_price == 3000.0
        assert user.preferences.preferred_locations == ["Pacific Heights"]
    
    def test_deactivate_user(self):
        """Test user deactivation."""
        user = self.user_factory.create()
        assert user.is_active is True
        
        user.deactivate()
        assert user.is_active is False
    
    def test_activate_user(self):
        """Test user activation."""
        user = self.user_factory.create()
        user.deactivate()
        assert user.is_active is False
        
        user.activate()
        assert user.is_active is True
    
    def test_budget_conscious_user_factory(self):
        """Test budget-conscious user factory method."""
        user = self.user_factory.create_budget_conscious_user()
        
        assert user.preferences.min_price == 800
        assert user.preferences.max_price == 2500
        assert user.preferences.max_bedrooms == 2
        assert "laundry" in user.preferences.required_amenities
        assert "studio" in user.preferences.property_types
        assert "apartment" in user.preferences.property_types
    
    def test_luxury_seeker_user_factory(self):
        """Test luxury-seeking user factory method."""
        user = self.user_factory.create_luxury_seeker_user()
        
        assert user.preferences.min_price == 5000
        assert user.preferences.max_price == 20000
        assert user.preferences.min_bedrooms == 2
        assert "parking" in user.preferences.required_amenities
        assert "gym" in user.preferences.required_amenities
        assert "concierge" in user.preferences.required_amenities
        assert "luxury_apartment" in user.preferences.property_types
    
    def test_user_with_interactions_factory(self):
        """Test user creation with interactions factory method."""
        properties = self.property_factory.create_batch(10)
        user = self.user_factory.create_user_with_interactions(
            properties=properties,
            interaction_count=5
        )
        
        assert len(user.interactions) == 5
        
        # Check that interactions are with different properties
        interacted_property_ids = [interaction.property_id for interaction in user.interactions]
        assert len(set(interacted_property_ids)) == 5  # All different properties
        
        # Check that all interacted properties are from the provided list
        property_ids = [prop.id for prop in properties]
        assert all(prop_id in property_ids for prop_id in interacted_property_ids)
    
    def test_user_batch_creation(self):
        """Test creating multiple users."""
        users = self.user_factory.create_batch(5)
        
        assert len(users) == 5
        assert all(isinstance(user, User) for user in users)
        
        # Check that all users have unique IDs and emails
        user_ids = [user.id for user in users]
        user_emails = [user.email for user in users]
        
        assert len(set(user_ids)) == 5
        assert len(set(user_emails)) == 5
    
    def test_user_interaction_chronology(self):
        """Test that interactions maintain chronological order."""
        user = self.user_factory.create()
        property_id = uuid4()
        
        # Add interactions with slight delays
        interaction1 = UserInteraction.create(property_id, "view")
        user.add_interaction(interaction1)
        
        # Small delay to ensure different timestamps
        import time
        time.sleep(0.01)
        
        interaction2 = UserInteraction.create(property_id, "like")
        user.add_interaction(interaction2)
        
        time.sleep(0.01)
        
        interaction3 = UserInteraction.create(property_id, "save")
        user.add_interaction(interaction3)
        
        interactions = user.get_interaction_history()
        assert len(interactions) == 3
        
        # Check chronological order
        assert interactions[0].timestamp <= interactions[1].timestamp
        assert interactions[1].timestamp <= interactions[2].timestamp
    
    def test_user_created_at_timestamp(self):
        """Test that created_at timestamp is recent."""
        user = self.user_factory.create()
        
        time_diff = datetime.now() - user.created_at
        assert time_diff.total_seconds() < 5  # Should be created within 5 seconds
    
    def test_user_multiple_interactions_same_property(self):
        """Test user can have multiple interactions with same property."""
        user = self.user_factory.create()
        property_id = uuid4()
        
        # Add multiple interactions with same property
        interactions = [
            UserInteraction.create(property_id, "view", 60),
            UserInteraction.create(property_id, "like"),
            UserInteraction.create(property_id, "save"),
            UserInteraction.create(property_id, "inquiry")
        ]
        
        for interaction in interactions:
            user.add_interaction(interaction)
        
        assert len(user.interactions) == 4
        
        # All interactions should be with the same property
        assert all(interaction.property_id == property_id for interaction in user.interactions)
        
        # But should have different types
        interaction_types = [interaction.interaction_type for interaction in user.interactions]
        assert len(set(interaction_types)) == 4
    
    def test_user_email_validation(self):
        """Test user email format validation."""
        # Note: Current implementation doesn't enforce email validation
        # This test documents current behavior
        
        valid_emails = [
            "test@example.com",
            "user.name@domain.co.uk",
            "test+tag@example.org"
        ]
        
        for email in valid_emails:
            user = User.create(email=email)
            assert user.email == email
        
        # Test with invalid email formats (currently allowed)
        invalid_emails = [
            "notanemail",
            "@domain.com",
            "user@",
            ""
        ]
        
        for email in invalid_emails:
            user = User.create(email=email)
            assert user.email == email  # Currently no validation
    
    def test_user_preferences_edge_cases(self):
        """Test edge cases for user preferences."""
        # Extremely high prices
        high_price_prefs = UserPreferences(
            min_price=100000.0,
            max_price=500000.0
        )
        user = User.create("test@example.com", high_price_prefs)
        assert user.preferences.min_price == 100000.0
        
        # Zero prices
        zero_price_prefs = UserPreferences(
            min_price=0.0,
            max_price=0.0
        )
        user = User.create("test@example.com", zero_price_prefs)
        assert user.preferences.min_price == 0.0
        
        # Inverted price range (min > max)
        inverted_prefs = UserPreferences(
            min_price=5000.0,
            max_price=2000.0  # Less than min
        )
        user = User.create("test@example.com", inverted_prefs)
        assert user.preferences.min_price == 5000.0
        assert user.preferences.max_price == 2000.0  # Currently allowed
        
        # Large number of locations
        many_locations = [f"Location_{i}" for i in range(100)]
        many_locations_prefs = UserPreferences(preferred_locations=many_locations)
        user = User.create("test@example.com", many_locations_prefs)
        assert len(user.preferences.preferred_locations) == 100
    
    def test_user_interaction_edge_cases(self):
        """Test edge cases for user interactions."""
        user = self.user_factory.create()
        property_id = uuid4()
        
        # Very long duration
        long_interaction = UserInteraction.create(
            property_id, "view", duration_seconds=86400  # 24 hours
        )
        user.add_interaction(long_interaction)
        assert user.interactions[0].duration_seconds == 86400
        
        # Zero duration
        zero_duration = UserInteraction.create(
            property_id, "view", duration_seconds=0
        )
        user.add_interaction(zero_duration)
        assert user.interactions[1].duration_seconds == 0
        
        # Negative duration (currently allowed)
        negative_duration = UserInteraction.create(
            property_id, "view", duration_seconds=-60
        )
        user.add_interaction(negative_duration)
        assert user.interactions[2].duration_seconds == -60
        
        # Custom interaction type
        custom_interaction = UserInteraction.create(
            property_id, "custom_action"
        )
        user.add_interaction(custom_interaction)
        assert user.interactions[3].interaction_type == "custom_action"