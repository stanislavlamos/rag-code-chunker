from abc import ABC, abstractmethod
from typing import List

class BaseChunker(ABC):
    """Base class for text chunkers."""
    
    @abstractmethod
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks."""
        pass