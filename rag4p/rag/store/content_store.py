from abc import ABC, abstractmethod
from typing import List

from rag4p.rag.model.chunk import Chunk


class ContentStore(ABC):

    @abstractmethod
    def store(self, chunks: List[Chunk]):
        pass
