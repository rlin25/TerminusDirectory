from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any
from uuid import UUID
import numpy as np


class ModelRepository(ABC):
    
    @abstractmethod
    async def save_model(self, model_name: str, model_data: Any, version: str) -> bool:
        pass
    
    @abstractmethod
    async def load_model(self, model_name: str, version: str = "latest") -> Optional[Any]:
        pass
    
    @abstractmethod
    async def get_model_versions(self, model_name: str) -> List[str]:
        pass
    
    @abstractmethod
    async def delete_model(self, model_name: str, version: str) -> bool:
        pass
    
    @abstractmethod
    async def save_embeddings(self, entity_type: str, entity_id: str, embeddings: np.ndarray) -> bool:
        pass
    
    @abstractmethod
    async def get_embeddings(self, entity_type: str, entity_id: str) -> Optional[np.ndarray]:
        pass
    
    @abstractmethod
    async def get_all_embeddings(self, entity_type: str) -> Dict[str, np.ndarray]:
        pass
    
    @abstractmethod
    async def save_training_metrics(self, model_name: str, version: str, metrics: Dict[str, float]) -> bool:
        pass
    
    @abstractmethod
    async def get_training_metrics(self, model_name: str, version: str) -> Optional[Dict[str, float]]:
        pass
    
    @abstractmethod
    async def cache_predictions(self, cache_key: str, predictions: Any, ttl_seconds: int = 3600) -> bool:
        pass
    
    @abstractmethod
    async def get_cached_predictions(self, cache_key: str) -> Optional[Any]:
        pass
    
    @abstractmethod
    async def clear_cache(self, pattern: str = "*") -> bool:
        pass