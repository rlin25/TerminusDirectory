from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime
from uuid import UUID, uuid4


@dataclass
class UserPreferences:
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    min_bedrooms: Optional[int] = None
    max_bedrooms: Optional[int] = None
    min_bathrooms: Optional[float] = None
    max_bathrooms: Optional[float] = None
    preferred_locations: List[str] = None
    required_amenities: List[str] = None
    property_types: List[str] = None
    
    def __post_init__(self):
        if self.preferred_locations is None:
            self.preferred_locations = []
        if self.required_amenities is None:
            self.required_amenities = []
        if self.property_types is None:
            self.property_types = ["apartment"]


@dataclass
class UserInteraction:
    property_id: UUID
    interaction_type: str  # "view", "like", "inquiry", "save"
    timestamp: datetime
    duration_seconds: Optional[int] = None
    
    @classmethod
    def create(cls, property_id: UUID, interaction_type: str, duration_seconds: Optional[int] = None):
        return cls(
            property_id=property_id,
            interaction_type=interaction_type,
            timestamp=datetime.now(),
            duration_seconds=duration_seconds
        )


@dataclass
class User:
    id: UUID
    email: str
    preferences: UserPreferences
    interactions: List[UserInteraction]
    created_at: datetime
    is_active: bool = True
    
    @classmethod
    def create(cls, email: str, preferences: UserPreferences = None):
        return cls(
            id=uuid4(),
            email=email,
            preferences=preferences or UserPreferences(),
            interactions=[],
            created_at=datetime.now()
        )
    
    def add_interaction(self, interaction: UserInteraction):
        self.interactions.append(interaction)
    
    def get_interaction_history(self, interaction_type: str = None) -> List[UserInteraction]:
        if interaction_type:
            return [i for i in self.interactions if i.interaction_type == interaction_type]
        return self.interactions
    
    def get_liked_properties(self) -> List[UUID]:
        return [i.property_id for i in self.interactions if i.interaction_type == "like"]
    
    def get_viewed_properties(self) -> List[UUID]:
        return [i.property_id for i in self.interactions if i.interaction_type == "view"]
    
    def update_preferences(self, new_preferences: UserPreferences):
        self.preferences = new_preferences
    
    def deactivate(self):
        self.is_active = False
    
    def activate(self):
        self.is_active = True