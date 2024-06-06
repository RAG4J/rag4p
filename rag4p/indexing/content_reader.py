from abc import ABC, abstractmethod
from typing import List, Iterable

from rag4p.indexing.input_document import InputDocument


class ContentReader(ABC):
    @abstractmethod
    def read(self, batch_size: int = 10) -> Iterable[List[InputDocument]]:
        """
        Read input documents in batches.

        Parameters:
        batch_size (int): The number of documents in each chunk.

        Returns:
        Iterable[List[InputDocument]]: An iterable of lists of input documents.
        """
        pass

    def name(self) -> str:
        return self.__class__.__name__
