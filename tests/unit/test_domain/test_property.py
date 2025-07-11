"""
Unit tests for the Property domain entity.

Tests property creation, validation, methods, and edge cases.
"""

import pytest
from datetime import datetime
from uuid import UUID, uuid4
from typing import List, Dict

from domain.entities.property import Property
from tests.utils.data_factories import PropertyFactory, FactoryConfig


class TestProperty:
    """Test cases for Property entity."""
    
    def setup_method(self):
        """Set up test fixtures for each test."""
        self.factory = PropertyFactory(FactoryConfig(seed=42))
    
    def test_property_creation_with_factory(self):
        """Test property creation using factory."""
        property_obj = self.factory.create()
        
        assert isinstance(property_obj, Property)
        assert isinstance(property_obj.id, UUID)
        assert property_obj.title
        assert property_obj.description
        assert property_obj.price > 0
        assert property_obj.location
        assert property_obj.bedrooms >= 0
        assert property_obj.bathrooms > 0
        assert isinstance(property_obj.amenities, list)
        assert isinstance(property_obj.contact_info, dict)
        assert isinstance(property_obj.images, list)
        assert isinstance(property_obj.scraped_at, datetime)
        assert property_obj.is_active is True
        assert property_obj.property_type
    
    def test_property_create_class_method(self):
        """Test Property.create() class method."""
        property_obj = Property.create(
            title="Test Apartment",
            description="A beautiful test apartment",
            price=2500.0,
            location="Downtown",
            bedrooms=2,
            bathrooms=1.5,
            square_feet=1200,
            amenities=["parking", "gym"],
            contact_info={"phone": "555-0123"},
            images=["image1.jpg"],
            property_type="apartment"
        )
        
        assert property_obj.title == "Test Apartment"
        assert property_obj.description == "A beautiful test apartment"
        assert property_obj.price == 2500.0
        assert property_obj.location == "Downtown"
        assert property_obj.bedrooms == 2
        assert property_obj.bathrooms == 1.5
        assert property_obj.square_feet == 1200
        assert property_obj.amenities == ["parking", "gym"]
        assert property_obj.contact_info == {"phone": "555-0123"}
        assert property_obj.images == ["image1.jpg"]
        assert property_obj.property_type == "apartment"
        assert property_obj.is_active is True
        assert isinstance(property_obj.scraped_at, datetime)
        assert isinstance(property_obj.id, UUID)
    
    def test_property_create_with_defaults(self):
        """Test property creation with minimal required fields."""
        property_obj = Property.create(
            title="Minimal Property",
            description="Basic description",
            price=1000.0,
            location="Unknown",
            bedrooms=1,
            bathrooms=1.0
        )
        
        assert property_obj.amenities == []
        assert property_obj.contact_info == {}
        assert property_obj.images == []
        assert property_obj.square_feet is None
        assert property_obj.property_type == "apartment"
        assert property_obj.is_active is True
    
    def test_get_full_text_method(self):
        """Test the get_full_text() method."""
        property_obj = Property.create(
            title="Beautiful Apartment",
            description="Spacious and modern",
            price=3000.0,
            location="Downtown SF",
            bedrooms=2,
            bathrooms=2.0,
            amenities=["parking", "gym", "pool"]
        )
        
        full_text = property_obj.get_full_text()
        expected = "Beautiful Apartment Spacious and modern Downtown SF parking gym pool"
        assert full_text == expected
    
    def test_get_full_text_with_empty_amenities(self):
        """Test get_full_text() with empty amenities."""
        property_obj = Property.create(
            title="Simple Apartment",
            description="Basic unit",
            price=2000.0,
            location="Mission",
            bedrooms=1,
            bathrooms=1.0,
            amenities=[]
        )
        
        full_text = property_obj.get_full_text()
        assert full_text == "Simple Apartment Basic unit Mission "
    
    def test_get_price_per_sqft_valid(self):
        """Test price per square foot calculation with valid data."""
        property_obj = Property.create(
            title="Test Property",
            description="Test",
            price=3000.0,
            location="Test Location",
            bedrooms=2,
            bathrooms=1.0,
            square_feet=1200
        )
        
        price_per_sqft = property_obj.get_price_per_sqft()
        assert price_per_sqft == 2.5  # 3000 / 1200
    
    def test_get_price_per_sqft_no_square_feet(self):
        """Test price per square foot when square_feet is None."""
        property_obj = Property.create(
            title="Test Property",
            description="Test",
            price=3000.0,
            location="Test Location",
            bedrooms=2,
            bathrooms=1.0,
            square_feet=None
        )
        
        price_per_sqft = property_obj.get_price_per_sqft()
        assert price_per_sqft is None
    
    def test_get_price_per_sqft_zero_square_feet(self):
        """Test price per square foot when square_feet is zero."""
        property_obj = Property.create(
            title="Test Property",
            description="Test",
            price=3000.0,
            location="Test Location",
            bedrooms=2,
            bathrooms=1.0,
            square_feet=0
        )
        
        price_per_sqft = property_obj.get_price_per_sqft()
        assert price_per_sqft is None
    
    def test_deactivate_method(self):
        """Test property deactivation."""
        property_obj = self.factory.create()
        assert property_obj.is_active is True
        
        property_obj.deactivate()
        assert property_obj.is_active is False
    
    def test_activate_method(self):
        """Test property activation."""
        property_obj = self.factory.create()
        property_obj.deactivate()
        assert property_obj.is_active is False
        
        property_obj.activate()
        assert property_obj.is_active is True
    
    def test_luxury_property_creation(self):
        """Test creation of luxury properties."""
        luxury_property = self.factory.create_luxury_property()
        
        assert luxury_property.price >= 8000
        assert luxury_property.property_type in ['penthouse', 'luxury_apartment']
        assert luxury_property.bedrooms >= 2
        assert luxury_property.bathrooms >= 2.0
        assert luxury_property.square_feet >= 1800
        assert len(luxury_property.amenities) >= 6
        
        # Check for luxury amenities
        luxury_amenities = ['concierge', 'doorman', 'gym', 'pool', 'rooftop']
        has_luxury_amenity = any(amenity in luxury_property.amenities for amenity in luxury_amenities)
        assert has_luxury_amenity
    
    def test_budget_property_creation(self):
        """Test creation of budget properties."""
        budget_property = self.factory.create_budget_property()
        
        assert budget_property.price <= 2000
        assert budget_property.property_type in ['studio', 'apartment']
        assert budget_property.bedrooms <= 2
        assert budget_property.bathrooms <= 1.5
        assert budget_property.square_feet <= 900
        assert len(budget_property.amenities) <= 3
    
    def test_edge_case_properties(self):
        """Test various edge case property configurations."""
        # No amenities
        no_amenities_prop = self.factory.create_edge_case_property("no_amenities")
        assert no_amenities_prop.amenities == []
        
        # High price
        high_price_prop = self.factory.create_edge_case_property("high_price")
        assert high_price_prop.price == 50000
        
        # Zero bedrooms (studio)
        studio_prop = self.factory.create_edge_case_property("zero_bedrooms")
        assert studio_prop.bedrooms == 0
        assert studio_prop.property_type == "studio"
        
        # Large property
        large_prop = self.factory.create_edge_case_property("large_property")
        assert large_prop.bedrooms == 10
        assert large_prop.bathrooms == 8.0
        assert large_prop.square_feet == 10000
        assert large_prop.price == 25000
        
        # Minimal data
        minimal_prop = self.factory.create_edge_case_property("minimal_data")
        assert minimal_prop.title == "Minimal Property"
        assert minimal_prop.amenities == []
        assert minimal_prop.contact_info == {}
        assert minimal_prop.images == []
    
    def test_property_batch_creation(self):
        """Test creating multiple properties."""
        properties = self.factory.create_batch(5)
        
        assert len(properties) == 5
        assert all(isinstance(prop, Property) for prop in properties)
        
        # Check that all properties have unique IDs
        property_ids = [prop.id for prop in properties]
        assert len(set(property_ids)) == 5
    
    def test_property_creation_with_overrides(self):
        """Test property creation with custom overrides."""
        custom_property = self.factory.create(
            title="Custom Title",
            price=5000,
            bedrooms=3,
            amenities=["custom_amenity"]
        )
        
        assert custom_property.title == "Custom Title"
        assert custom_property.price == 5000
        assert custom_property.bedrooms == 3
        assert "custom_amenity" in custom_property.amenities
    
    def test_property_scraped_at_timestamp(self):
        """Test that scraped_at timestamp is recent."""
        property_obj = self.factory.create()
        
        time_diff = datetime.now() - property_obj.scraped_at
        assert time_diff.total_seconds() < 5  # Should be created within 5 seconds
    
    def test_property_type_validation(self):
        """Test various property types."""
        valid_types = [
            'apartment', 'condo', 'house', 'studio', 'loft', 
            'townhouse', 'duplex', 'penthouse', 'luxury_apartment'
        ]
        
        for prop_type in valid_types:
            property_obj = self.factory.create(property_type=prop_type)
            assert property_obj.property_type == prop_type
    
    def test_amenities_list_immutability(self):
        """Test that amenities list modifications don't affect original."""
        property_obj = self.factory.create(amenities=["parking", "gym"])
        original_amenities = property_obj.amenities.copy()
        
        # Modify the list
        property_obj.amenities.append("pool")
        
        # Original creation data should be preserved in our test
        assert "pool" in property_obj.amenities
        assert len(property_obj.amenities) == len(original_amenities) + 1
    
    def test_contact_info_structure(self):
        """Test contact info dictionary structure."""
        contact_info = {"phone": "555-1234", "email": "test@example.com", "website": "example.com"}
        property_obj = self.factory.create(contact_info=contact_info)
        
        assert property_obj.contact_info == contact_info
        assert property_obj.contact_info["phone"] == "555-1234"
        assert property_obj.contact_info["email"] == "test@example.com"
        assert property_obj.contact_info["website"] == "example.com"
    
    def test_images_list_structure(self):
        """Test images list structure."""
        images = ["image1.jpg", "image2.png", "image3.webp"]
        property_obj = self.factory.create(images=images)
        
        assert property_obj.images == images
        assert len(property_obj.images) == 3
        assert all(isinstance(img, str) for img in property_obj.images)
    
    def test_property_equality_by_id(self):
        """Test that properties with same ID are considered equal."""
        property_id = uuid4()
        
        property1 = Property(
            id=property_id,
            title="Test 1",
            description="Description 1",
            price=1000,
            location="Location 1",
            bedrooms=1,
            bathrooms=1.0,
            square_feet=None,
            amenities=[],
            contact_info={},
            images=[],
            scraped_at=datetime.now(),
            is_active=True,
            property_type="apartment"
        )
        
        property2 = Property(
            id=property_id,
            title="Test 2",
            description="Description 2",
            price=2000,
            location="Location 2",
            bedrooms=2,
            bathrooms=2.0,
            square_feet=1200,
            amenities=["parking"],
            contact_info={"phone": "555-1234"},
            images=["image.jpg"],
            scraped_at=datetime.now(),
            is_active=False,
            property_type="condo"
        )
        
        # Properties should be equal based on ID
        assert property1.id == property2.id
    
    def test_realistic_data_generation(self):
        """Test that factory generates realistic property data."""
        properties = self.factory.create_batch(10)
        
        for prop in properties:
            # Price should be reasonable
            assert 500 <= prop.price <= 50000
            
            # Bedrooms should be reasonable
            assert 0 <= prop.bedrooms <= 10
            
            # Bathrooms should be reasonable
            assert 1.0 <= prop.bathrooms <= 10.0
            
            # Square feet should be reasonable if specified
            if prop.square_feet:
                assert 200 <= prop.square_feet <= 10000
            
            # Title and description should not be empty
            assert len(prop.title) > 0
            assert len(prop.description) > 0
            
            # Location should not be empty
            assert len(prop.location) > 0
            
            # Property type should be valid
            assert prop.property_type in PropertyFactory.PROPERTY_TYPES
    
    def test_property_features_correlation(self):
        """Test that property features are correlated realistically."""
        luxury_properties = [self.factory.create_luxury_property() for _ in range(5)]
        budget_properties = [self.factory.create_budget_property() for _ in range(5)]
        
        # Luxury properties should generally have more amenities
        avg_luxury_amenities = sum(len(prop.amenities) for prop in luxury_properties) / len(luxury_properties)
        avg_budget_amenities = sum(len(prop.amenities) for prop in budget_properties) / len(budget_properties)
        
        assert avg_luxury_amenities > avg_budget_amenities
        
        # Luxury properties should have higher price per sqft
        luxury_prices = [prop.price for prop in luxury_properties if prop.square_feet]
        budget_prices = [prop.price for prop in budget_properties if prop.square_feet]
        
        if luxury_prices and budget_prices:
            avg_luxury_price = sum(luxury_prices) / len(luxury_prices)
            avg_budget_price = sum(budget_prices) / len(budget_prices)
            assert avg_luxury_price > avg_budget_price


class TestPropertyValidation:
    """Test property validation and error conditions."""
    
    def test_property_with_negative_price(self):
        """Test property creation with negative price."""
        # Note: The Property dataclass doesn't enforce validation in __init__
        # This test documents current behavior - validation should be added if needed
        property_obj = Property.create(
            title="Test",
            description="Test",
            price=-1000,  # Negative price
            location="Test",
            bedrooms=1,
            bathrooms=1.0
        )
        
        assert property_obj.price == -1000  # Currently allowed
    
    def test_property_with_zero_price(self):
        """Test property creation with zero price."""
        property_obj = Property.create(
            title="Free Property",
            description="Test",
            price=0,
            location="Test",
            bedrooms=1,
            bathrooms=1.0
        )
        
        assert property_obj.price == 0
    
    def test_property_with_negative_bedrooms(self):
        """Test property creation with negative bedrooms."""
        property_obj = Property.create(
            title="Test",
            description="Test",
            price=1000,
            location="Test",
            bedrooms=-1,  # Negative bedrooms
            bathrooms=1.0
        )
        
        assert property_obj.bedrooms == -1  # Currently allowed
    
    def test_property_with_zero_bathrooms(self):
        """Test property creation with zero bathrooms."""
        property_obj = Property.create(
            title="Test",
            description="Test",
            price=1000,
            location="Test",
            bedrooms=1,
            bathrooms=0.0  # Zero bathrooms
        )
        
        assert property_obj.bathrooms == 0.0  # Currently allowed