from abc import ABC, abstractmethod


class Embedder(ABC):
    """
    Abstract class for embedding text.
    """
    @abstractmethod
    def embed(self, text: str) -> [float]:
        pass
