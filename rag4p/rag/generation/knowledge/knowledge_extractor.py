from abc import ABC, abstractmethod
from typing import List

from rag4p.rag.generation.knowledge.knowledge import Knowledge


class KnowledgeExtractor(ABC):
    """
    Abstract class for extracting knowledge from a context.
    """
    @abstractmethod
    def extract_knowledge(self, context: str) -> List[Knowledge]:
        pass
