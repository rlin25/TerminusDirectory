from dataclasses import dataclass
from typing import List, Dict, Optional
from datetime import datetime
from uuid import UUID, uuid4


@dataclass
class SearchFilters:
    min_price: Optional[float] = None
    max_price: Optional[float] = None
    min_bedrooms: Optional[int] = None
    max_bedrooms: Optional[int] = None
    min_bathrooms: Optional[float] = None
    max_bathrooms: Optional[float] = None
    locations: List[str] = None
    amenities: List[str] = None
    property_types: List[str] = None
    min_square_feet: Optional[int] = None
    max_square_feet: Optional[int] = None
    
    def __post_init__(self):
        if self.locations is None:
            self.locations = []
        if self.amenities is None:
            self.amenities = []
        if self.property_types is None:
            self.property_types = []


@dataclass
class SearchQuery:
    id: UUID
    user_id: Optional[UUID]
    query_text: str
    filters: SearchFilters
    created_at: datetime
    limit: int = 50
    offset: int = 0
    sort_by: str = "relevance"  # "relevance", "price_asc", "price_desc", "date_new", "date_old"
    
    @classmethod
    def create(cls, query_text: str, user_id: Optional[UUID] = None, 
               filters: SearchFilters = None, limit: int = 50, offset: int = 0,
               sort_by: str = "relevance"):
        return cls(
            id=uuid4(),
            user_id=user_id,
            query_text=query_text,
            filters=filters or SearchFilters(),
            created_at=datetime.now(),
            limit=limit,
            offset=offset,
            sort_by=sort_by
        )
    
    def get_normalized_query(self) -> str:
        return self.query_text.lower().strip()
    
    def has_location_filter(self) -> bool:
        return bool(self.filters.locations)
    
    def has_price_filter(self) -> bool:
        return self.filters.min_price is not None or self.filters.max_price is not None
    
    def has_size_filter(self) -> bool:
        return (self.filters.min_bedrooms is not None or 
                self.filters.max_bedrooms is not None or
                self.filters.min_bathrooms is not None or
                self.filters.max_bathrooms is not None)
    
    def to_dict(self) -> Dict:
        return {
            "id": str(self.id),
            "user_id": str(self.user_id) if self.user_id else None,
            "query_text": self.query_text,
            "filters": {
                "min_price": self.filters.min_price,
                "max_price": self.filters.max_price,
                "min_bedrooms": self.filters.min_bedrooms,
                "max_bedrooms": self.filters.max_bedrooms,
                "min_bathrooms": self.filters.min_bathrooms,
                "max_bathrooms": self.filters.max_bathrooms,
                "locations": self.filters.locations,
                "amenities": self.filters.amenities,
                "property_types": self.filters.property_types,
                "min_square_feet": self.filters.min_square_feet,
                "max_square_feet": self.filters.max_square_feet
            },
            "created_at": self.created_at.isoformat(),
            "limit": self.limit,
            "offset": self.offset,
            "sort_by": self.sort_by
        }
    
    def get_cache_key(self) -> str:
        """Generate a cache key for this search query"""
        import hashlib
        import json
        
        # Create a normalized representation for hashing
        cache_data = {
            "query": self.query_text.lower().strip(),
            "filters": {
                "min_price": self.filters.min_price,
                "max_price": self.filters.max_price,
                "min_bedrooms": self.filters.min_bedrooms,
                "max_bedrooms": self.filters.max_bedrooms,
                "min_bathrooms": self.filters.min_bathrooms,
                "max_bathrooms": self.filters.max_bathrooms,
                "locations": sorted(self.filters.locations) if self.filters.locations else [],
                "amenities": sorted(self.filters.amenities) if self.filters.amenities else [],
                "property_types": sorted(self.filters.property_types) if self.filters.property_types else [],
                "min_square_feet": self.filters.min_square_feet,
                "max_square_feet": self.filters.max_square_feet
            },
            "limit": self.limit,
            "offset": self.offset,
            "sort_by": self.sort_by
        }
        
        # Convert to JSON string and hash
        cache_string = json.dumps(cache_data, sort_keys=True)
        return hashlib.md5(cache_string.encode('utf-8')).hexdigest()