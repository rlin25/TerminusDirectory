from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime
from uuid import UUID, uuid4


@dataclass
class Property:
    id: UUID
    title: str
    description: str
    price: float
    location: str
    bedrooms: int
    bathrooms: float
    square_feet: Optional[int]
    amenities: List[str]
    contact_info: Dict[str, str]
    images: List[str]
    scraped_at: datetime
    is_active: bool = True
    property_type: str = "apartment"
    
    @classmethod
    def create(cls, title: str, description: str, price: float, location: str, 
               bedrooms: int, bathrooms: float, square_feet: Optional[int] = None,
               amenities: List[str] = None, contact_info: Dict[str, str] = None,
               images: List[str] = None, property_type: str = "apartment"):
        return cls(
            id=uuid4(),
            title=title,
            description=description,
            price=price,
            location=location,
            bedrooms=bedrooms,
            bathrooms=bathrooms,
            square_feet=square_feet,
            amenities=amenities or [],
            contact_info=contact_info or {},
            images=images or [],
            scraped_at=datetime.now(),
            property_type=property_type
        )
    
    def get_full_text(self) -> str:
        return f"{self.title} {self.description} {self.location} {' '.join(self.amenities)}"
    
    def get_price_per_sqft(self) -> Optional[float]:
        if self.square_feet and self.square_feet > 0:
            return self.price / self.square_feet
        return None
    
    def deactivate(self):
        self.is_active = False
    
    def activate(self):
        self.is_active = True