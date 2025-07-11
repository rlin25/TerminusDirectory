"""
Sample Data Generator for Rental ML System Demo

This module generates realistic sample data for demonstration purposes:
- Properties with varied characteristics
- Users with different preferences
- Realistic interactions and behavior patterns
- ML training data simulation
"""

import random
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any
from uuid import uuid4
import json

from src.domain.entities.property import Property
from src.domain.entities.user import User, UserPreferences, UserInteraction


class SampleDataGenerator:
    """Generates realistic sample data for the demo application"""
    
    def __init__(self, seed: int = 42):
        """Initialize the data generator with a random seed"""
        random.seed(seed)
        np.random.seed(seed)
        
        # Define realistic data pools
        self.locations = [
            "Downtown", "Midtown", "Uptown", "Financial District", 
            "Arts District", "Suburban Heights", "Riverside", "Hillside",
            "University Area", "Old Town", "Tech Quarter", "Green Valley",
            "Sunset Park", "Brookside", "Central Plaza", "Marina District"
        ]
        
        self.property_types = ["apartment", "house", "condo", "studio", "townhouse"]
        
        self.amenities_pool = [
            "parking", "gym", "pool", "laundry", "pet-friendly", "balcony",
            "air-conditioning", "heating", "dishwasher", "hardwood-floors",
            "in-unit-laundry", "walk-in-closet", "fireplace", "garden",
            "rooftop-access", "concierge", "security", "storage"
        ]
        
        self.property_titles = [
            "Luxury Downtown Apartment", "Cozy Studio Retreat", "Modern Family Home",
            "Spacious Loft", "Charming Cottage", "Executive Condo", "Garden Apartment",
            "Penthouse Suite", "Riverside Townhouse", "Urban Oasis", "Historic Apartment",
            "Contemporary Living Space", "Designer Studio", "Family-Friendly Home",
            "Elegant Condo", "Artistic Loft", "Quiet Retreat", "City View Apartment"
        ]
        
        self.description_templates = [
            "Beautiful {property_type} featuring {amenities}. Located in the heart of {location}.",
            "Stunning {property_type} with modern amenities including {amenities}. Perfect for {target_audience}.",
            "Newly renovated {property_type} in {location}. Features include {amenities}.",
            "Spacious {property_type} offering {amenities}. Convenient location in {location}.",
            "Charming {property_type} with {amenities}. Great neighborhood in {location}."
        ]
        
        self.target_audiences = [
            "professionals", "students", "families", "couples", "young professionals",
            "retirees", "artists", "remote workers"
        ]
        
        self.user_domains = [
            "gmail.com", "yahoo.com", "outlook.com", "icloud.com", 
            "company.com", "university.edu", "freelancer.com"
        ]
        
        self.first_names = [
            "John", "Jane", "Michael", "Sarah", "David", "Emily", "Robert", "Lisa",
            "Christopher", "Jennifer", "Matthew", "Ashley", "Joshua", "Amanda",
            "Daniel", "Jessica", "Andrew", "Brittany", "James", "Samantha"
        ]
        
        self.last_names = [
            "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller",
            "Davis", "Rodriguez", "Martinez", "Hernandez", "Lopez", "Gonzalez",
            "Wilson", "Anderson", "Thomas", "Taylor", "Moore", "Jackson", "Martin"
        ]

    def generate_properties(self, count: int = 100) -> List[Property]:
        """Generate a list of realistic property objects"""
        properties = []
        
        for _ in range(count):
            # Basic property characteristics
            property_type = random.choice(self.property_types)
            location = random.choice(self.locations)
            
            # Bedrooms based on property type
            if property_type == "studio":
                bedrooms = 0
            elif property_type == "apartment":
                bedrooms = random.choices([1, 2, 3, 4], weights=[30, 40, 25, 5])[0]
            elif property_type == "house":
                bedrooms = random.choices([2, 3, 4, 5], weights=[20, 40, 30, 10])[0]
            elif property_type == "condo":
                bedrooms = random.choices([1, 2, 3], weights=[25, 50, 25])[0]
            else:  # townhouse
                bedrooms = random.choices([2, 3, 4], weights=[30, 50, 20])[0]
            
            # Bathrooms (correlated with bedrooms)
            if bedrooms == 0:
                bathrooms = 1.0
            elif bedrooms == 1:
                bathrooms = random.choices([1.0, 1.5], weights=[70, 30])[0]
            elif bedrooms == 2:
                bathrooms = random.choices([1.0, 1.5, 2.0], weights=[20, 30, 50])[0]
            elif bedrooms == 3:
                bathrooms = random.choices([1.5, 2.0, 2.5], weights=[20, 60, 20])[0]
            else:
                bathrooms = random.choices([2.0, 2.5, 3.0], weights=[40, 40, 20])[0]
            
            # Square footage (correlated with bedrooms and property type)
            base_sqft = {
                "studio": 400,
                "apartment": 600,
                "condo": 750,
                "townhouse": 1000,
                "house": 1200
            }
            
            sqft_base = base_sqft[property_type]
            sqft_per_bedroom = 300
            square_feet = int(sqft_base + (bedrooms * sqft_per_bedroom) + random.normalvariate(0, 150))
            square_feet = max(300, square_feet)  # Minimum 300 sq ft
            
            # Price calculation (location and size based)
            location_multipliers = {
                "Downtown": 1.4, "Financial District": 1.5, "Tech Quarter": 1.3,
                "Midtown": 1.2, "Arts District": 1.1, "Marina District": 1.4,
                "Uptown": 1.0, "University Area": 0.9, "Central Plaza": 1.2,
                "Suburban Heights": 0.8, "Riverside": 1.1, "Hillside": 0.9,
                "Old Town": 1.0, "Green Valley": 0.7, "Sunset Park": 0.8,
                "Brookside": 0.75
            }
            
            base_price_per_sqft = random.normalvariate(2.5, 0.3)
            location_mult = location_multipliers[location]
            price = square_feet * base_price_per_sqft * location_mult
            
            # Add property type adjustment
            type_adjustments = {
                "studio": 0.9, "apartment": 1.0, "condo": 1.1,
                "townhouse": 1.05, "house": 1.15
            }
            price *= type_adjustments[property_type]
            
            # Round to nearest $50
            price = round(price / 50) * 50
            price = max(800, price)  # Minimum $800
            
            # Generate amenities
            num_amenities = random.choices([2, 3, 4, 5, 6], weights=[10, 30, 35, 20, 5])[0]
            amenities = random.sample(self.amenities_pool, num_amenities)
            
            # Title and description
            title = random.choice(self.property_titles)
            if property_type in title.lower():
                # Title already mentions property type
                pass
            else:
                title = f"{title} {property_type.title()}"
            
            description_template = random.choice(self.description_templates)
            target_audience = random.choice(self.target_audiences)
            amenities_text = ", ".join(amenities[:3])
            
            description = description_template.format(
                property_type=property_type,
                location=location,
                amenities=amenities_text,
                target_audience=target_audience
            )
            
            # Contact information
            contact_info = {
                "phone": f"({random.randint(200, 999)}) {random.randint(200, 999)}-{random.randint(1000, 9999)}",
                "email": f"contact@property{random.randint(1000, 9999)}.com"
            }
            
            # Images (placeholder URLs)
            num_images = random.randint(3, 8)
            images = [f"https://example.com/property_{uuid4().hex[:8]}_{i}.jpg" for i in range(num_images)]
            
            # Create property
            property_obj = Property.create(
                title=title,
                description=description,
                price=price,
                location=location,
                bedrooms=bedrooms,
                bathrooms=bathrooms,
                square_feet=square_feet,
                amenities=amenities,
                contact_info=contact_info,
                images=images,
                property_type=property_type
            )
            
            # Set scraped_at to a random recent time
            days_ago = random.randint(1, 30)
            property_obj.scraped_at = datetime.now() - timedelta(days=days_ago)
            
            # Randomly deactivate some properties
            if random.random() < 0.05:  # 5% inactive
                property_obj.deactivate()
            
            properties.append(property_obj)
        
        return properties

    def generate_users(self, count: int = 50) -> List[User]:
        """Generate a list of realistic user objects with preferences"""
        users = []
        
        for _ in range(count):
            # Generate email
            first_name = random.choice(self.first_names)
            last_name = random.choice(self.last_names)
            domain = random.choice(self.user_domains)
            
            username = f"{first_name.lower()}.{last_name.lower()}"
            if random.random() < 0.3:  # 30% chance of adding numbers
                username += str(random.randint(1, 999))
            
            email = f"{username}@{domain}"
            
            # Generate preferences
            preferences = self._generate_user_preferences()
            
            # Create user
            user = User.create(email=email, preferences=preferences)
            
            # Set creation date to a random time in the past
            days_ago = random.randint(1, 365)
            user.created_at = datetime.now() - timedelta(days=days_ago)
            
            users.append(user)
        
        return users

    def _generate_user_preferences(self) -> UserPreferences:
        """Generate realistic user preferences"""
        # Price preferences
        income_bracket = random.choices(
            ["low", "medium", "high", "luxury"], 
            weights=[25, 45, 25, 5]
        )[0]
        
        if income_bracket == "low":
            min_price = random.randint(800, 1200)
            max_price = random.randint(min_price + 200, 2000)
        elif income_bracket == "medium":
            min_price = random.randint(1500, 2500)
            max_price = random.randint(min_price + 500, 4000)
        elif income_bracket == "high":
            min_price = random.randint(3000, 4500)
            max_price = random.randint(min_price + 1000, 8000)
        else:  # luxury
            min_price = random.randint(6000, 8000)
            max_price = random.randint(min_price + 2000, 15000)
        
        # Bedroom preferences
        bedroom_pref = random.choices(
            ["studio", "small", "medium", "large"],
            weights=[15, 35, 40, 10]
        )[0]
        
        if bedroom_pref == "studio":
            min_bedrooms, max_bedrooms = 0, 1
        elif bedroom_pref == "small":
            min_bedrooms, max_bedrooms = 1, 2
        elif bedroom_pref == "medium":
            min_bedrooms, max_bedrooms = 2, 3
        else:  # large
            min_bedrooms, max_bedrooms = 3, 5
        
        # Bathroom preferences
        min_bathrooms = random.choices([1.0, 1.5, 2.0], weights=[50, 30, 20])[0]
        max_bathrooms = min_bathrooms + random.choices([0, 0.5, 1.0], weights=[40, 40, 20])[0]
        
        # Location preferences
        num_preferred_locations = random.choices([0, 1, 2, 3], weights=[30, 40, 25, 5])[0]
        preferred_locations = []
        if num_preferred_locations > 0:
            preferred_locations = random.sample(self.locations, num_preferred_locations)
        
        # Amenity preferences
        num_required_amenities = random.choices([0, 1, 2, 3, 4], weights=[20, 35, 30, 12, 3])[0]
        required_amenities = []
        if num_required_amenities > 0:
            # More likely to require common amenities
            common_amenities = ["parking", "laundry", "air-conditioning", "heating"]
            other_amenities = [a for a in self.amenities_pool if a not in common_amenities]
            
            # Pick some common ones first
            common_picked = min(num_required_amenities, len(common_amenities))
            required_amenities.extend(random.sample(common_amenities, min(2, common_picked)))
            
            # Fill remaining with other amenities
            remaining = num_required_amenities - len(required_amenities)
            if remaining > 0:
                required_amenities.extend(random.sample(other_amenities, min(remaining, len(other_amenities))))
        
        # Property type preferences
        property_type_pref = random.choices(
            ["any", "apartments_only", "houses_only", "specific"],
            weights=[40, 30, 15, 15]
        )[0]
        
        if property_type_pref == "any":
            property_types = self.property_types.copy()
        elif property_type_pref == "apartments_only":
            property_types = ["apartment", "studio", "condo"]
        elif property_type_pref == "houses_only":
            property_types = ["house", "townhouse"]
        else:  # specific
            num_types = random.choices([1, 2, 3], weights=[50, 35, 15])[0]
            property_types = random.sample(self.property_types, num_types)
        
        return UserPreferences(
            min_price=min_price,
            max_price=max_price,
            min_bedrooms=min_bedrooms,
            max_bedrooms=max_bedrooms,
            min_bathrooms=min_bathrooms,
            max_bathrooms=max_bathrooms,
            preferred_locations=preferred_locations,
            required_amenities=required_amenities,
            property_types=property_types
        )

    def generate_interactions(self, users: List[User], properties: List[Property], count: int = 500) -> List[UserInteraction]:
        """Generate realistic user interactions with properties"""
        interactions = []
        interaction_types = ["view", "like", "inquiry", "save"]
        type_weights = [60, 20, 10, 10]  # Views are most common
        
        for _ in range(count):
            user = random.choice(users)
            property_obj = random.choice(properties)
            interaction_type = random.choices(interaction_types, weights=type_weights)[0]
            
            # Duration for view interactions
            duration = None
            if interaction_type == "view":
                # Log-normal distribution for viewing time
                duration = max(5, int(np.random.lognormal(3, 1)))  # 5 seconds to several minutes
            
            # Create interaction
            interaction = UserInteraction.create(
                property_id=property_obj.id,
                interaction_type=interaction_type,
                duration_seconds=duration
            )
            
            # Set timestamp to a random time in the past
            days_ago = random.randint(0, 90)
            hours_ago = random.randint(0, 23)
            minutes_ago = random.randint(0, 59)
            
            interaction.timestamp = datetime.now() - timedelta(
                days=days_ago, 
                hours=hours_ago, 
                minutes=minutes_ago
            )
            
            # Add interaction to user
            user.add_interaction(interaction)
            interactions.append(interaction)
        
        return interactions

    def generate_ml_training_data(self, users: List[User], properties: List[Property]) -> Dict[str, Any]:
        """Generate ML training data structures"""
        
        # Create user-item interaction matrix
        user_ids = [user.id for user in users]
        property_ids = [prop.id for prop in properties]
        
        # Create mapping dictionaries
        user_to_idx = {user_id: idx for idx, user_id in enumerate(user_ids)}
        property_to_idx = {prop_id: idx for idx, prop_id in enumerate(property_ids)}
        
        # Initialize interaction matrix
        interaction_matrix = np.zeros((len(users), len(properties)))
        
        # Fill interaction matrix based on user interactions
        for user in users:
            user_idx = user_to_idx[user.id]
            
            for interaction in user.interactions:
                if interaction.property_id in property_to_idx:
                    prop_idx = property_to_idx[interaction.property_id]
                    
                    # Weight different interaction types
                    weights = {"view": 1, "like": 3, "save": 2, "inquiry": 4}
                    weight = weights.get(interaction.interaction_type, 1)
                    
                    interaction_matrix[user_idx, prop_idx] += weight
        
        # Normalize to 0-5 scale (like ratings)
        max_interaction = interaction_matrix.max()
        if max_interaction > 0:
            interaction_matrix = (interaction_matrix / max_interaction) * 5
        
        # Generate property features matrix
        property_features = []
        for prop in properties:
            features = self._extract_property_features(prop)
            property_features.append(features)
        
        property_features_matrix = np.array(property_features)
        
        # Generate user features matrix
        user_features = []
        for user in users:
            features = self._extract_user_features(user)
            user_features.append(features)
        
        user_features_matrix = np.array(user_features)
        
        return {
            "user_item_matrix": interaction_matrix,
            "property_features": property_features_matrix,
            "user_features": user_features_matrix,
            "user_mapping": user_to_idx,
            "property_mapping": property_to_idx,
            "feature_names": {
                "property": self._get_property_feature_names(),
                "user": self._get_user_feature_names()
            }
        }

    def _extract_property_features(self, property_obj: Property) -> List[float]:
        """Extract numerical features from a property"""
        features = []
        
        # Basic numerical features
        features.append(property_obj.price)
        features.append(property_obj.bedrooms)
        features.append(property_obj.bathrooms)
        features.append(property_obj.square_feet or 0)
        features.append(property_obj.get_price_per_sqft() or 0)
        
        # Location one-hot encoding
        for location in self.locations:
            features.append(1.0 if property_obj.location == location else 0.0)
        
        # Property type one-hot encoding
        for prop_type in self.property_types:
            features.append(1.0 if property_obj.property_type == prop_type else 0.0)
        
        # Amenity features (binary)
        for amenity in self.amenities_pool:
            features.append(1.0 if amenity in property_obj.amenities else 0.0)
        
        # Derived features
        features.append(len(property_obj.amenities))  # Total amenity count
        features.append(len(property_obj.images))      # Image count
        features.append(1.0 if property_obj.is_active else 0.0)  # Active status
        
        # Time-based features
        days_since_scraped = (datetime.now() - property_obj.scraped_at).days
        features.append(days_since_scraped)
        
        return features

    def _extract_user_features(self, user: User) -> List[float]:
        """Extract numerical features from a user"""
        features = []
        prefs = user.preferences
        
        # Preference features
        features.append(prefs.min_price or 0)
        features.append(prefs.max_price or 10000)
        features.append((prefs.max_price or 10000) - (prefs.min_price or 0))  # Price range
        features.append(prefs.min_bedrooms or 0)
        features.append(prefs.max_bedrooms or 5)
        features.append(prefs.min_bathrooms or 1.0)
        features.append(prefs.max_bathrooms or 3.0)
        
        # Location preferences (count)
        features.append(len(prefs.preferred_locations))
        
        # Amenity preferences (count)
        features.append(len(prefs.required_amenities))
        
        # Property type preferences (count)
        features.append(len(prefs.property_types))
        
        # User activity features
        features.append(len(user.interactions))
        features.append(len(user.get_liked_properties()))
        features.append(len(user.get_viewed_properties()))
        
        # Time-based features
        days_since_joined = (datetime.now() - user.created_at).days
        features.append(days_since_joined)
        
        # Interaction type distributions
        total_interactions = len(user.interactions)
        if total_interactions > 0:
            view_ratio = len([i for i in user.interactions if i.interaction_type == "view"]) / total_interactions
            like_ratio = len([i for i in user.interactions if i.interaction_type == "like"]) / total_interactions
            inquiry_ratio = len([i for i in user.interactions if i.interaction_type == "inquiry"]) / total_interactions
            save_ratio = len([i for i in user.interactions if i.interaction_type == "save"]) / total_interactions
        else:
            view_ratio = like_ratio = inquiry_ratio = save_ratio = 0.0
        
        features.extend([view_ratio, like_ratio, inquiry_ratio, save_ratio])
        
        return features

    def _get_property_feature_names(self) -> List[str]:
        """Get names of property features in order"""
        names = [
            "price", "bedrooms", "bathrooms", "square_feet", "price_per_sqft"
        ]
        
        # Location features
        names.extend([f"location_{loc.replace(' ', '_').lower()}" for loc in self.locations])
        
        # Property type features
        names.extend([f"type_{ptype}" for ptype in self.property_types])
        
        # Amenity features
        names.extend([f"amenity_{amenity.replace('-', '_')}" for amenity in self.amenities_pool])
        
        # Derived features
        names.extend([
            "amenity_count", "image_count", "is_active", "days_since_scraped"
        ])
        
        return names

    def _get_user_feature_names(self) -> List[str]:
        """Get names of user features in order"""
        return [
            "min_price", "max_price", "price_range",
            "min_bedrooms", "max_bedrooms", "min_bathrooms", "max_bathrooms",
            "preferred_locations_count", "required_amenities_count", "property_types_count",
            "total_interactions", "liked_properties_count", "viewed_properties_count",
            "days_since_joined", "view_ratio", "like_ratio", "inquiry_ratio", "save_ratio"
        ]

    def generate_performance_metrics(self) -> Dict[str, Any]:
        """Generate sample ML model performance metrics"""
        
        # Generate time series data for the last 30 days
        dates = [datetime.now() - timedelta(days=i) for i in range(30)]
        dates.reverse()
        
        metrics = {
            "accuracy": {
                "collaborative_filtering": [0.85 + random.normalvariate(0, 0.02) for _ in dates],
                "content_based": [0.80 + random.normalvariate(0, 0.015) for _ in dates],
                "hybrid": [0.90 + random.normalvariate(0, 0.01) for _ in dates],
                "dates": [d.strftime("%Y-%m-%d") for d in dates]
            },
            "response_times": {
                "avg_response_ms": [50 + random.normalvariate(0, 10) for _ in range(24)],
                "95th_percentile_ms": [90 + random.normalvariate(0, 15) for _ in range(24)],
                "hours": list(range(24))
            },
            "user_engagement": {
                "click_through_rate": [0.12 + random.normalvariate(0, 0.01) for _ in dates],
                "conversion_rate": [0.037 + random.normalvariate(0, 0.003) for _ in dates],
                "user_satisfaction": [4.5 + random.normalvariate(0, 0.1) for _ in dates],
                "dates": [d.strftime("%Y-%m-%d") for d in dates]
            },
            "system_health": {
                "api_success_rate": [0.995 + random.normalvariate(0, 0.002) for _ in dates],
                "scraping_success_rate": [0.985 + random.normalvariate(0, 0.005) for _ in dates],
                "database_performance": [25 + random.normalvariate(0, 3) for _ in dates],  # ms
                "dates": [d.strftime("%Y-%m-%d") for d in dates]
            }
        }
        
        return metrics

    def export_sample_data(self, filename: str, users: List[User], properties: List[Property], interactions: List[UserInteraction]) -> None:
        """Export sample data to JSON file for persistence"""
        
        data = {
            "users": [
                {
                    "id": str(user.id),
                    "email": user.email,
                    "created_at": user.created_at.isoformat(),
                    "preferences": {
                        "min_price": user.preferences.min_price,
                        "max_price": user.preferences.max_price,
                        "min_bedrooms": user.preferences.min_bedrooms,
                        "max_bedrooms": user.preferences.max_bedrooms,
                        "min_bathrooms": user.preferences.min_bathrooms,
                        "max_bathrooms": user.preferences.max_bathrooms,
                        "preferred_locations": user.preferences.preferred_locations,
                        "required_amenities": user.preferences.required_amenities,
                        "property_types": user.preferences.property_types
                    }
                }
                for user in users
            ],
            "properties": [
                {
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
                    "scraped_at": prop.scraped_at.isoformat(),
                    "is_active": prop.is_active,
                    "property_type": prop.property_type
                }
                for prop in properties
            ],
            "interactions": [
                {
                    "property_id": str(interaction.property_id),
                    "interaction_type": interaction.interaction_type,
                    "timestamp": interaction.timestamp.isoformat(),
                    "duration_seconds": interaction.duration_seconds
                }
                for interaction in interactions
            ]
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)