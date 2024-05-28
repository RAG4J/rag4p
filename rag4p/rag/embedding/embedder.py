from abc import ABC, abstractmethod


class Embedder(ABC):
    """
    Abstract class for embedding text.
    """
    @abstractmethod
    def embed(self, text: str) -> [float]:
        pass

    @abstractmethod
    def identifier(self) -> str:
        pass

    @staticmethod
    def supplier() -> str:
        pass

    @abstractmethod
    def model(self) -> str:
        pass
