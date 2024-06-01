from abc import abstractmethod, ABC
from typing import List

from rag4p.rag.model.chunk import Chunk
from rag4p.indexing.input_document import InputDocument


class Splitter(ABC):

    @abstractmethod
    def split(self, input_document: InputDocument) -> List[Chunk]:
        pass

    @staticmethod
    @abstractmethod
    def name() -> str:
        pass
    