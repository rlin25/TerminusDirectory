#!/usr/bin/env python3
"""
Simple test script to verify PostgreSQL Property Repository implementation.
This script tests the basic functionality without requiring a full test setup.
"""

import asyncio
import os
import sys
from datetime import datetime
from uuid import uuid4

# Add the source directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from domain.entities.property import Property
from domain.entities.search_query import SearchQuery, SearchFilters
from infrastructure.data.repositories.postgres_property_repository import PostgresPropertyRepository


async def main():
    """Test the PostgreSQL Property Repository implementation."""
    
    # Database URL for testing (this would come from environment variables in real usage)
    database_url = "postgresql+asyncpg://user:password@localhost:5432/test_db"
    
    # Initialize repository
    repo = PostgresPropertyRepository(database_url)
    
    print("PostgreSQL Property Repository Implementation Test")
    print("=" * 50)
    
    try:
        # Test 1: Health check
        print("1. Testing health check...")
        health_status = await repo.health_check()
        print(f"   Health status: {health_status['status']}")
        
        # Test 2: Create a test property
        print("\n2. Testing property creation...")
        test_property = Property.create(
            title="Test Apartment",
            description="A beautiful test apartment in downtown",
            price=2500.0,
            location="San Francisco, CA",
            bedrooms=2,
            bathrooms=2.0,
            square_feet=1000,
            amenities=["parking", "gym", "pool"],
            contact_info={"phone": "555-0123", "email": "test@example.com"},
            images=["image1.jpg", "image2.jpg"],
            property_type="apartment"
        )
        
        created_property = await repo.create(test_property)
        print(f"   Created property with ID: {created_property.id}")
        
        # Test 3: Get property by ID
        print("\n3. Testing get property by ID...")
        retrieved_property = await repo.get_by_id(created_property.id)
        if retrieved_property:
            print(f"   Retrieved property: {retrieved_property.title}")
            print(f"   Price: ${retrieved_property.price}")
            print(f"   Active: {retrieved_property.is_active}")
        else:
            print("   Property not found!")
        
        # Test 4: Search properties
        print("\n4. Testing property search...")
        search_query = SearchQuery.create(
            query_text="apartment",
            filters=SearchFilters(
                min_price=2000.0,
                max_price=3000.0,
                min_bedrooms=2
            ),
            limit=10
        )
        
        search_results, total_count = await repo.search(search_query)
        print(f"   Found {len(search_results)} properties out of {total_count} total")
        
        # Test 5: Get property features
        print("\n5. Testing get property features...")
        features = await repo.get_property_features(created_property.id)
        if features:
            print(f"   Property features extracted: {len(features)} fields")
            print(f"   Data quality score: {features.get('data_quality_score', 'N/A')}")
        
        # Test 6: Update property
        print("\n6. Testing property update...")
        created_property.price = 2700.0
        created_property.description = "Updated beautiful test apartment"
        updated_property = await repo.update(created_property)
        print(f"   Updated property price to: ${updated_property.price}")
        
        # Test 7: Get property counts
        print("\n7. Testing property counts...")
        total_count = await repo.get_count()
        active_count = await repo.get_active_count()
        print(f"   Total properties: {total_count}")
        print(f"   Active properties: {active_count}")
        
        # Test 8: Soft delete property
        print("\n8. Testing property deletion...")
        deleted = await repo.delete(created_property.id)
        print(f"   Property deleted: {deleted}")
        
        # Test 9: Verify property is inactive
        deleted_property = await repo.get_by_id(created_property.id)
        if deleted_property is None:
            print("   Confirmed: Property no longer retrievable (soft deleted)")
        
        print("\n" + "=" * 50)
        print("All tests completed successfully!")
        
    except Exception as e:
        print(f"\nError during testing: {e}")
        print(f"Error type: {type(e).__name__}")
        import traceback
        traceback.print_exc()
        
    finally:
        # Cleanup
        await repo.close()
        print("\nRepository connections closed.")


def test_domain_model_compatibility():
    """Test that our domain model works correctly with the repository model conversion."""
    print("\nTesting Domain Model Compatibility")
    print("-" * 40)
    
    # Create a property using the domain model
    property = Property.create(
        title="Domain Test Property",
        description="Testing domain model compatibility",
        price=1500.0,
        location="Test City, TX",
        bedrooms=1,
        bathrooms=1.0,
        square_feet=800,
        amenities=["air_conditioning", "dishwasher"],
        property_type="studio"
    )
    
    print(f"Created domain property: {property.title}")
    print(f"Property ID: {property.id}")
    print(f"Is active: {property.is_active}")
    print(f"Price per sqft: {property.get_price_per_sqft()}")
    print(f"Full text: {property.get_full_text()[:100]}...")
    
    # Test the PropertyModel conversion
    from infrastructure.data.repositories.postgres_property_repository import PropertyModel
    
    model = PropertyModel.from_domain(property)
    print(f"\nConverted to database model:")
    print(f"Status: {model.status}")
    print(f"Data quality score: {model.data_quality_score}")
    
    converted_back = model.to_domain()
    print(f"\nConverted back to domain:")
    print(f"Is active: {converted_back.is_active}")
    print(f"Title matches: {converted_back.title == property.title}")
    print(f"Price matches: {converted_back.price == property.price}")
    
    print("Domain model compatibility test completed!")


if __name__ == "__main__":
    # First test domain model compatibility (no database needed)
    test_domain_model_compatibility()
    
    # Note: The full async test requires a database connection
    print("\n" + "=" * 60)
    print("NOTE: Full repository test requires database connection.")
    print("To run the full test, ensure PostgreSQL is running and update")
    print("the database URL in the script, then uncomment the line below.")
    print("=" * 60)
    
    # Uncomment the next line to run the full test with database:
    # asyncio.run(main())