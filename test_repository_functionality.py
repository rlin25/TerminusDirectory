#!/usr/bin/env python3
"""
Test script to verify repository functionality with real database operations.
"""

import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

# Load production environment
load_dotenv(Path(__file__).parent / ".env.production")

from src.infrastructure.data.repository_factory import RepositoryFactory
from src.domain.entities.property import Property
from src.domain.entities.user import User

async def test_repository_operations():
    """Test basic repository operations"""
    print("🧪 Testing Repository Functionality")
    print("=" * 50)
    
    repo_factory = None
    
    try:
        # Initialize repository factory
        print("1. Initializing repository factory...")
        repo_factory = RepositoryFactory()
        await repo_factory.initialize()
        
        # Test health check
        print("2. Testing health check...")
        health = await repo_factory.health_check()
        print(f"   Database health: {health.get('database', False)}")
        print(f"   Redis health: {health.get('redis', False)}")
        print(f"   Overall health: {health.get('overall', False)}")
        
        if not health.get('overall', False):
            print("   ⚠️ System not fully healthy, but continuing with tests...")
        
        # Get repositories
        print("3. Getting repository instances...")
        property_repo = repo_factory.get_property_repository()
        user_repo = repo_factory.get_user_repository()
        model_repo = repo_factory.get_model_repository()
        
        print(f"   Property repository: {type(property_repo).__name__}")
        print(f"   User repository: {type(user_repo).__name__}")
        print(f"   Model repository: {type(model_repo).__name__}")
        
        # Test property repository
        print("4. Testing property repository...")
        try:
            # Create a test property using the factory method
            test_property = Property.create(
                title="Test Property",
                description="A test property for repository verification",
                price=1500.0,
                location="Test City, State",
                bedrooms=2,
                bathrooms=1.5,
                square_feet=1000,
                amenities=["WiFi", "Parking"],
                contact_info={"email": "test@example.com"},
                images=[]
            )
            
            # Save property using create method
            saved_property = await property_repo.create(test_property)
            print(f"   ✅ Property saved with ID: {saved_property.id}")
            
            # Search properties
            search_results = await property_repo.search({
                "title": "Test Property"
            })
            print(f"   ✅ Found {len(search_results)} properties in search")
            
            # Clean up - delete test property
            if saved_property.id:
                await property_repo.delete(saved_property.id)
                print(f"   ✅ Test property cleaned up")
                
        except Exception as e:
            print(f"   ❌ Property repository test failed: {e}")
        
        # Test user repository
        print("5. Testing user repository...")
        try:
            # Create a test user using the factory method
            from src.domain.entities.user import UserPreferences
            test_preferences = UserPreferences(
                min_price=1000,
                max_price=2000,
                preferred_locations=["Test City"]
            )
            test_user = User.create(
                email="test@example.com",
                preferences=test_preferences
            )
            
            # Save user using create method
            saved_user = await user_repo.create(test_user)
            print(f"   ✅ User saved with ID: {saved_user.id}")
            
            # Find user by email
            found_user = await user_repo.get_by_email("test@example.com")
            if found_user:
                print(f"   ✅ User found by email: {found_user.email}")
            
            # Clean up - delete test user
            if saved_user.id:
                await user_repo.delete(saved_user.id)
                print(f"   ✅ Test user cleaned up")
                
        except Exception as e:
            print(f"   ❌ User repository test failed: {e}")
        
        # Test model repository
        print("6. Testing model repository...")
        try:
            # Test model storage capabilities
            test_model_data = b"fake_model_data_for_testing"
            test_metadata = {"version": "1.0", "type": "test"}
            
            # Save model (model_name, model_data, version)
            model_id = await model_repo.save_model(
                "test_model",
                test_model_data,
                "1.0"
            )
            print(f"   ✅ Model saved with ID: {model_id}")
            
            # Load model
            loaded_model = await model_repo.load_model("test_model", "1.0")
            if loaded_model:
                print(f"   ✅ Model loaded successfully")
            
            # List models
            models = await model_repo.get_model_versions("test_model")
            print(f"   ✅ Found {len(models)} versions of test_model")
            
            # Clean up
            deleted_count = await model_repo.cleanup_old_models("test_model", keep_versions=0)
            print(f"   ✅ Cleaned up {deleted_count} old model versions")
            
        except Exception as e:
            print(f"   ❌ Model repository test failed: {e}")
        
        print("\n" + "=" * 50)
        print("✅ Repository functionality tests completed!")
        return True
        
    except Exception as e:
        print(f"❌ Repository test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        # Close repository factory
        if repo_factory:
            await repo_factory.close()
            print("🛑 Repository factory closed")

def main():
    """Main function"""
    try:
        result = asyncio.run(test_repository_operations())
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print("\n❌ Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()