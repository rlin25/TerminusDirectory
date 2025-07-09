from abc import ABC, abstractmethod
from typing import List, Optional
from uuid import UUID

from ..entities.user import User, UserInteraction


class UserRepository(ABC):
    
    @abstractmethod
    async def create(self, user: User) -> User:
        pass
    
    @abstractmethod
    async def get_by_id(self, user_id: UUID) -> Optional[User]:
        pass
    
    @abstractmethod
    async def get_by_email(self, email: str) -> Optional[User]:
        pass
    
    @abstractmethod
    async def update(self, user: User) -> User:
        pass
    
    @abstractmethod
    async def delete(self, user_id: UUID) -> bool:
        pass
    
    @abstractmethod
    async def get_all_active(self, limit: int = 100, offset: int = 0) -> List[User]:
        pass
    
    @abstractmethod
    async def add_interaction(self, user_id: UUID, interaction: UserInteraction) -> bool:
        pass
    
    @abstractmethod
    async def get_interactions(self, user_id: UUID, interaction_type: str = None, 
                             limit: int = 100, offset: int = 0) -> List[UserInteraction]:
        pass
    
    @abstractmethod
    async def get_user_interaction_matrix(self) -> dict:
        pass
    
    @abstractmethod
    async def get_users_who_liked_property(self, property_id: UUID) -> List[User]:
        pass
    
    @abstractmethod
    async def get_similar_users(self, user_id: UUID, limit: int = 10) -> List[User]:
        pass
    
    @abstractmethod
    async def get_count(self) -> int:
        pass
    
    @abstractmethod
    async def get_active_count(self) -> int:
        pass