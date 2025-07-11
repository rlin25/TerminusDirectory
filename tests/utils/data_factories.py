"""
Data factories for generating test data for the rental ML system.

This module provides factory classes for creating consistent test data
across different test modules.
"""

import random
import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta
from uuid import uuid4, UUID
from dataclasses import dataclass

from domain.entities.property import Property
from domain.entities.user import User, UserPreferences, UserInteraction
from domain.entities.search_query import SearchQuery


@dataclass
class FactoryConfig:
    """Configuration for data factories."""
    seed: int = 42
    use_realistic_data: bool = True
    include_edge_cases: bool = False


class PropertyFactory:
    """Factory for creating Property entities."""
    
    NEIGHBORHOODS = [
        'Downtown', 'Mission', 'SoMa', 'Castro', 'Marina', 'Richmond', 
        'Sunset', 'Haight', 'Nob Hill', 'Pacific Heights', 'Presidio',
        'Chinatown', 'Financial District', 'Union Square', 'Tenderloin'
    ]
    
    PROPERTY_TYPES = [
        'apartment', 'condo', 'house', 'studio', 'loft', 'townhouse',
        'duplex', 'penthouse', 'luxury_apartment'
    ]
    
    AMENITIES_POOL = [
        'parking', 'gym', 'pool', 'rooftop', 'laundry', 'dishwasher',
        'balcony', 'garden', 'elevator', 'concierge', 'doorman',
        'pet_friendly', 'air_conditioning', 'hardwood_floors',
        'in_unit_laundry', 'walk_in_closet', 'fireplace', 'view',
        'terrace', 'jacuzzi', 'sauna', 'bike_storage', 'storage_unit'
    ]
    
    def __init__(self, config: FactoryConfig = None):
        self.config = config or FactoryConfig()
        np.random.seed(self.config.seed)
        random.seed(self.config.seed)
    
    def create(self, **kwargs) -> Property:
        """Create a single Property with optional overrides."""
        defaults = self._generate_property_data()
        defaults.update(kwargs)
        
        return Property.create(
            title=defaults['title'],
            description=defaults['description'],
            price=defaults['price'],
            location=defaults['location'],
            bedrooms=defaults['bedrooms'],
            bathrooms=defaults['bathrooms'],
            square_feet=defaults.get('square_feet'),
            amenities=defaults['amenities'],
            contact_info=defaults['contact_info'],
            images=defaults['images'],
            property_type=defaults['property_type']
        )
    
    def create_batch(self, count: int, **common_kwargs) -> List[Property]:
        """Create multiple properties with optional common attributes."""
        properties = []
        for i in range(count):
            property_kwargs = common_kwargs.copy()
            # Add index to make each property unique
            if 'title' not in property_kwargs:
                property_kwargs['title'] = f"Test Property {i+1}"
            
            properties.append(self.create(**property_kwargs))
        
        return properties
    
    def create_luxury_property(self, **kwargs) -> Property:
        """Create a luxury property with high-end features."""
        luxury_defaults = {
            'price': random.uniform(8000, 15000),
            'property_type': random.choice(['penthouse', 'luxury_apartment']),
            'bedrooms': random.randint(2, 4),
            'bathrooms': random.uniform(2.0, 4.0),
            'square_feet': random.randint(1800, 3500),
            'amenities': random.sample([
                'concierge', 'doorman', 'gym', 'pool', 'rooftop', 
                'parking', 'view', 'terrace', 'jacuzzi', 'sauna'
            ], k=random.randint(6, 10))
        }
        luxury_defaults.update(kwargs)
        return self.create(**luxury_defaults)
    
    def create_budget_property(self, **kwargs) -> Property:
        """Create a budget-friendly property."""
        budget_defaults = {
            'price': random.uniform(800, 2000),
            'property_type': random.choice(['studio', 'apartment']),
            'bedrooms': random.randint(0, 2),
            'bathrooms': random.uniform(1.0, 1.5),
            'square_feet': random.randint(400, 900),
            'amenities': random.sample([
                'laundry', 'parking', 'pet_friendly'
            ], k=random.randint(1, 3))
        }
        budget_defaults.update(kwargs)
        return self.create(**budget_defaults)
    
    def create_edge_case_property(self, case_type: str, **kwargs) -> Property:
        """Create properties for edge case testing."""
        if case_type == "no_amenities":
            kwargs.update({'amenities': []})
        elif case_type == "high_price":
            kwargs.update({'price': 50000})
        elif case_type == "zero_bedrooms":
            kwargs.update({'bedrooms': 0, 'property_type': 'studio'})
        elif case_type == "large_property":
            kwargs.update({
                'bedrooms': 10, 
                'bathrooms': 8.0, 
                'square_feet': 10000,
                'price': 25000
            })
        elif case_type == "minimal_data":
            # Only required fields
            return Property.create(
                title="Minimal Property",
                description="Basic description",
                price=1000,
                location="Unknown",
                bedrooms=1,
                bathrooms=1.0
            )
        
        return self.create(**kwargs)
    
    def _generate_property_data(self) -> Dict[str, Any]:
        """Generate realistic property data."""
        neighborhood = random.choice(self.NEIGHBORHOODS)
        property_type = random.choice(self.PROPERTY_TYPES)
        
        # Generate correlated features
        if property_type == 'studio':
            bedrooms = 0
            bathrooms = 1.0
            square_feet = random.randint(300, 600)
        elif property_type in ['penthouse', 'luxury_apartment']:
            bedrooms = random.randint(2, 5)
            bathrooms = round(random.uniform(2.0, bedrooms + 1), 1)
            square_feet = random.randint(1500, 4000)
        else:
            bedrooms = random.randint(1, 4)
            bathrooms = round(random.uniform(1.0, min(bedrooms + 1, 3.5)), 1)
            square_feet = random.randint(600, 2500)
        
        # Price calculation with location premium
        location_premiums = {
            'Downtown': 1.4, 'SoMa': 1.5, 'Mission': 1.2, 'Castro': 1.3,
            'Marina': 1.6, 'Pacific Heights': 1.8, 'Nob Hill': 1.7,
            'Financial District': 1.5, 'Union Square': 1.4,
            'Richmond': 0.9, 'Sunset': 0.8, 'Haight': 1.0,
            'Tenderloin': 0.7, 'Chinatown': 0.9, 'Presidio': 1.3
        }
        
        base_price_per_sqft = random.uniform(2.0, 5.0)
        location_premium = location_premiums.get(neighborhood, 1.0)
        price = int(square_feet * base_price_per_sqft * location_premium)
        
        # Generate amenities based on property type and price
        amenity_count = min(len(self.AMENITIES_POOL), max(2, int(price / 1000)))
        amenities = random.sample(self.AMENITIES_POOL, k=random.randint(2, amenity_count))
        
        # Ensure luxury properties have luxury amenities
        if property_type in ['penthouse', 'luxury_apartment'] or price > 6000:
            luxury_amenities = ['concierge', 'doorman', 'gym', 'pool', 'rooftop']
            amenities.extend([a for a in luxury_amenities if a not in amenities])
        
        return {
            'title': f"Beautiful {bedrooms}BR {property_type.replace('_', ' ').title()} in {neighborhood}",
            'description': self._generate_description(neighborhood, property_type, amenities),
            'price': price,
            'location': f"{neighborhood}, San Francisco, CA",
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'square_feet': square_feet,
            'amenities': amenities,
            'contact_info': {
                'phone': f"555-{random.randint(1000, 9999)}",
                'email': f"contact_{uuid4().hex[:8]}@example.com"
            },
            'images': [f"image_{uuid4().hex[:8]}.jpg" for _ in range(random.randint(2, 6))],
            'property_type': property_type
        }
    
    def _generate_description(self, neighborhood: str, property_type: str, amenities: List[str]) -> str:
        """Generate a realistic property description."""
        templates = [
            f"Stunning {property_type.replace('_', ' ')} in the heart of {neighborhood}. Features include {', '.join(amenities[:3])} and much more!",
            f"Modern {property_type.replace('_', ' ')} with premium finishes located in desirable {neighborhood}. Enjoy {', '.join(amenities[:2])} and other great amenities.",
            f"Spacious {property_type.replace('_', ' ')} in {neighborhood} offering {', '.join(amenities[:4])}. Perfect for urban living!",
            f"Beautifully appointed {property_type.replace('_', ' ')} in prime {neighborhood} location. This unit boasts {', '.join(amenities[:3])}."
        ]
        return random.choice(templates)


class UserFactory:
    """Factory for creating User entities."""
    
    DOMAINS = ['gmail.com', 'yahoo.com', 'hotmail.com', 'example.com', 'test.com']
    
    def __init__(self, config: FactoryConfig = None):
        self.config = config or FactoryConfig()
        np.random.seed(self.config.seed)
        random.seed(self.config.seed)
    
    def create(self, **kwargs) -> User:
        """Create a single User with optional overrides."""
        defaults = self._generate_user_data()
        defaults.update(kwargs)
        
        preferences = defaults.get('preferences')
        if isinstance(preferences, dict):
            preferences = UserPreferences(**preferences)
        
        return User.create(
            email=defaults['email'],
            preferences=preferences
        )
    
    def create_batch(self, count: int, **common_kwargs) -> List[User]:
        """Create multiple users."""
        users = []
        for i in range(count):
            user_kwargs = common_kwargs.copy()
            if 'email' not in user_kwargs:
                user_kwargs['email'] = f"testuser{i}@example.com"
            users.append(self.create(**user_kwargs))
        return users
    
    def create_budget_conscious_user(self, **kwargs) -> User:
        """Create a user with budget-conscious preferences."""
        budget_preferences = UserPreferences(
            min_price=800,
            max_price=2500,
            min_bedrooms=0,
            max_bedrooms=2,
            preferred_locations=['Richmond', 'Sunset', 'Outer Mission'],
            required_amenities=['laundry'],
            property_types=['studio', 'apartment']
        )
        kwargs.update({'preferences': budget_preferences})
        return self.create(**kwargs)
    
    def create_luxury_seeker_user(self, **kwargs) -> User:
        """Create a user seeking luxury properties."""
        luxury_preferences = UserPreferences(
            min_price=5000,
            max_price=20000,
            min_bedrooms=2,
            max_bedrooms=5,
            preferred_locations=['Pacific Heights', 'Marina', 'SoMa'],
            required_amenities=['parking', 'gym', 'concierge'],
            property_types=['luxury_apartment', 'penthouse', 'condo']
        )
        kwargs.update({'preferences': luxury_preferences})
        return self.create(**kwargs)
    
    def create_user_with_interactions(self, properties: List[Property], 
                                    interaction_count: int = 5, **kwargs) -> User:
        """Create a user with pre-existing interactions."""
        user = self.create(**kwargs)
        
        # Add random interactions
        selected_properties = random.sample(properties, min(interaction_count, len(properties)))
        interaction_types = ['view', 'like', 'inquiry', 'save']
        
        for prop in selected_properties:
            interaction_type = random.choice(interaction_types)
            duration = random.randint(30, 300) if interaction_type == 'view' else None
            
            interaction = UserInteraction.create(
                property_id=prop.id,
                interaction_type=interaction_type,
                duration_seconds=duration
            )
            user.add_interaction(interaction)
        
        return user
    
    def _generate_user_data(self) -> Dict[str, Any]:
        """Generate realistic user data."""
        first_names = ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank', 'Grace', 'Henry']
        last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller']
        
        first_name = random.choice(first_names)
        last_name = random.choice(last_names)
        domain = random.choice(self.DOMAINS)
        
        email = f"{first_name.lower()}.{last_name.lower()}@{domain}"
        
        # Generate realistic preferences
        min_price = random.randint(1000, 3000)
        max_price = min_price + random.randint(1000, 5000)
        
        min_bedrooms = random.randint(0, 2)
        max_bedrooms = min_bedrooms + random.randint(1, 3)
        
        all_neighborhoods = PropertyFactory.NEIGHBORHOODS
        preferred_count = random.randint(1, 4)
        preferred_locations = random.sample(all_neighborhoods, preferred_count)
        
        all_amenities = PropertyFactory.AMENITIES_POOL[:10]  # Most common amenities
        required_count = random.randint(0, 3)
        required_amenities = random.sample(all_amenities, required_count)
        
        property_types = random.sample(PropertyFactory.PROPERTY_TYPES[:6], random.randint(1, 3))
        
        preferences = UserPreferences(
            min_price=min_price,
            max_price=max_price,
            min_bedrooms=min_bedrooms,
            max_bedrooms=max_bedrooms,
            preferred_locations=preferred_locations,
            required_amenities=required_amenities,
            property_types=property_types
        )
        
        return {
            'email': email,
            'preferences': preferences
        }


class SearchQueryFactory:
    """Factory for creating SearchQuery entities."""
    
    SEARCH_TERMS = [
        "cozy apartment", "luxury condo", "spacious house", "modern loft",
        "budget studio", "pet friendly", "gym parking", "downtown views",
        "quiet neighborhood", "walk to transit", "recently renovated"
    ]
    
    SORT_OPTIONS = ['price', 'price_desc', 'bedrooms', 'date_added', 'relevance']
    
    def __init__(self, config: FactoryConfig = None):
        self.config = config or FactoryConfig()
        random.seed(self.config.seed)
    
    def create(self, **kwargs) -> SearchQuery:
        """Create a single SearchQuery with optional overrides."""
        defaults = self._generate_search_data()
        defaults.update(kwargs)
        
        return SearchQuery.create(
            user_id=defaults['user_id'],
            query_text=defaults['query_text'],
            location=defaults.get('location'),
            min_price=defaults.get('min_price'),
            max_price=defaults.get('max_price'),
            bedrooms=defaults.get('bedrooms'),
            bathrooms=defaults.get('bathrooms'),
            amenities=defaults.get('amenities', []),
            property_type=defaults.get('property_type'),
            sort_by=defaults.get('sort_by', 'relevance')
        )
    
    def create_batch(self, count: int, **common_kwargs) -> List[SearchQuery]:
        """Create multiple search queries."""
        queries = []
        for i in range(count):
            query_kwargs = common_kwargs.copy()
            if 'user_id' not in query_kwargs:
                query_kwargs['user_id'] = uuid4()
            queries.append(self.create(**query_kwargs))
        return queries
    
    def create_specific_search(self, search_type: str, **kwargs) -> SearchQuery:
        """Create searches for specific scenarios."""
        if search_type == "budget_search":
            kwargs.update({
                'query_text': "affordable studio apartment",
                'max_price': 2000,
                'bedrooms': 0,
                'sort_by': 'price'
            })
        elif search_type == "luxury_search":
            kwargs.update({
                'query_text': "luxury penthouse with amenities",
                'min_price': 5000,
                'bedrooms': 3,
                'amenities': ['gym', 'concierge', 'pool'],
                'sort_by': 'price_desc'
            })
        elif search_type == "family_search":
            kwargs.update({
                'query_text': "family house with garden",
                'bedrooms': 3,
                'bathrooms': 2,
                'property_type': 'house',
                'amenities': ['parking', 'garden']
            })
        elif search_type == "empty_search":
            kwargs.update({
                'query_text': "",
                'location': None,
                'min_price': None,
                'max_price': None
            })
        
        return self.create(**kwargs)
    
    def _generate_search_data(self) -> Dict[str, Any]:
        """Generate realistic search query data."""
        query_text = random.choice(self.SEARCH_TERMS)
        location = random.choice(PropertyFactory.NEIGHBORHOODS) if random.random() > 0.3 else None
        
        # Price range (sometimes not specified)
        if random.random() > 0.4:
            min_price = random.randint(1000, 4000)
            max_price = min_price + random.randint(500, 3000) if random.random() > 0.3 else None
        else:
            min_price = max_price = None
        
        # Room specifications (sometimes not specified)
        bedrooms = random.randint(0, 4) if random.random() > 0.5 else None
        bathrooms = random.randint(1, 3) if random.random() > 0.7 else None
        
        # Amenities (sometimes specified)
        amenities = []
        if random.random() > 0.6:
            amenities = random.sample(PropertyFactory.AMENITIES_POOL[:8], random.randint(1, 3))
        
        # Property type (sometimes specified)
        property_type = random.choice(PropertyFactory.PROPERTY_TYPES) if random.random() > 0.5 else None
        
        return {
            'user_id': uuid4(),
            'query_text': query_text,
            'location': location,
            'min_price': min_price,
            'max_price': max_price,
            'bedrooms': bedrooms,
            'bathrooms': bathrooms,
            'amenities': amenities,
            'property_type': property_type,
            'sort_by': random.choice(self.SORT_OPTIONS)
        }


class MLDataFactory:
    """Factory for creating ML-specific test data."""
    
    def __init__(self, config: FactoryConfig = None):
        self.config = config or FactoryConfig()
        np.random.seed(self.config.seed)
    
    def create_user_item_matrix(self, num_users: int, num_items: int, 
                              density: float = 0.1) -> np.ndarray:
        """Create a realistic user-item interaction matrix."""
        matrix = np.zeros((num_users, num_items))
        
        # Generate interactions with realistic patterns
        for user_id in range(num_users):
            # User activity follows a power law distribution
            activity_level = np.random.pareto(1) + 1
            activity_level = min(activity_level, 10)  # Cap at 10x average
            
            num_interactions = int(num_items * density * activity_level / 2)
            num_interactions = min(num_interactions, num_items)
            
            if num_interactions > 0:
                # Users prefer certain property types (clustering)
                if user_id < num_users // 3:
                    # Budget users prefer lower indices (cheaper properties)
                    item_probs = np.exp(-np.arange(num_items) / (num_items / 3))
                elif user_id < 2 * num_users // 3:
                    # Middle-tier users prefer middle range
                    center = num_items // 2
                    item_probs = np.exp(-np.abs(np.arange(num_items) - center) / (num_items / 4))
                else:
                    # Luxury users prefer higher indices (expensive properties)
                    item_probs = np.exp(-np.arange(num_items)[::-1] / (num_items / 3))
                
                item_probs = item_probs / item_probs.sum()
                selected_items = np.random.choice(
                    num_items, size=num_interactions, replace=False, p=item_probs
                )
                
                # Set interaction strengths (binary for now, but could be ratings)
                matrix[user_id, selected_items] = 1
        
        return matrix
    
    def create_property_features(self, num_properties: int) -> List[Dict]:
        """Create property feature dictionaries for ML models."""
        property_factory = PropertyFactory(self.config)
        properties = property_factory.create_batch(num_properties)
        
        return [
            {
                'property_id': i,
                'neighborhood': prop.location.split(',')[0].strip(),
                'city': 'San Francisco',
                'price': prop.price,
                'bedrooms': prop.bedrooms,
                'bathrooms': prop.bathrooms,
                'square_feet': prop.square_feet or 1000,
                'amenities': prop.amenities,
                'property_type': prop.property_type
            }
            for i, prop in enumerate(properties)
        ]
    
    def create_training_data(self, num_users: int = 100, num_properties: int = 200,
                           density: float = 0.1) -> Dict[str, Any]:
        """Create complete training data for ML models."""
        user_item_matrix = self.create_user_item_matrix(num_users, num_properties, density)
        property_features = self.create_property_features(num_properties)
        
        return {
            'user_item_matrix': user_item_matrix,
            'property_features': property_features,
            'num_users': num_users,
            'num_properties': num_properties,
            'density': density
        }


# Factory registry for easy access
class FactoryRegistry:
    """Registry for all data factories."""
    
    def __init__(self, config: FactoryConfig = None):
        self.config = config or FactoryConfig()
        self.property_factory = PropertyFactory(self.config)
        self.user_factory = UserFactory(self.config)
        self.search_factory = SearchQueryFactory(self.config)
        self.ml_factory = MLDataFactory(self.config)
    
    def reset_seed(self, seed: int):
        """Reset the random seed for all factories."""
        self.config.seed = seed
        self.__init__(self.config)


# Global factory instance for easy import
default_factories = FactoryRegistry()