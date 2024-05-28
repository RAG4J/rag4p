from abc import ABC, abstractmethod
from types import MappingProxyType
from typing import List

from rag4p.rag.model.chunk import Chunk


class ContentStore(ABC):

    def __init__(self, metadata=None):
        if metadata is None:
            metadata = {}
        self._metadata = metadata

    def get_metadata(self):
        return MappingProxyType(self._metadata)

    def add_metadata(self, key: str, value):
        self._metadata[key] = value

    @abstractmethod
    def store(self, chunks: List[Chunk]):
        pass
