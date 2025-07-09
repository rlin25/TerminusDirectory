from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Tuple
from uuid import UUID

from ..entities.property import Property
from ..entities.search_query import SearchQuery


class PropertyRepository(ABC):
    
    @abstractmethod
    async def create(self, property: Property) -> Property:
        pass
    
    @abstractmethod
    async def get_by_id(self, property_id: UUID) -> Optional[Property]:
        pass
    
    @abstractmethod
    async def get_by_ids(self, property_ids: List[UUID]) -> List[Property]:
        pass
    
    @abstractmethod
    async def update(self, property: Property) -> Property:
        pass
    
    @abstractmethod
    async def delete(self, property_id: UUID) -> bool:
        pass
    
    @abstractmethod
    async def search(self, query: SearchQuery) -> Tuple[List[Property], int]:
        pass
    
    @abstractmethod
    async def get_all_active(self, limit: int = 100, offset: int = 0) -> List[Property]:
        pass
    
    @abstractmethod
    async def get_by_location(self, location: str, limit: int = 100, offset: int = 0) -> List[Property]:
        pass
    
    @abstractmethod
    async def get_by_price_range(self, min_price: float, max_price: float, 
                                limit: int = 100, offset: int = 0) -> List[Property]:
        pass
    
    @abstractmethod
    async def get_similar_properties(self, property_id: UUID, limit: int = 10) -> List[Property]:
        pass
    
    @abstractmethod
    async def bulk_create(self, properties: List[Property]) -> List[Property]:
        pass
    
    @abstractmethod
    async def get_property_features(self, property_id: UUID) -> Optional[Dict]:
        pass
    
    @abstractmethod
    async def update_property_features(self, property_id: UUID, features: Dict) -> bool:
        pass
    
    @abstractmethod
    async def get_count(self) -> int:
        pass
    
    @abstractmethod
    async def get_active_count(self) -> int:
        pass