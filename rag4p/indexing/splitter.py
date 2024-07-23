from abc import abstractmethod, ABC
from typing import List

from rag4p.rag.model.chunk import Chunk
from rag4p.indexing.input_document import InputDocument


class Splitter(ABC):
    """
    Abstract class for splitting an InputDocument or chunks into Chunks. The id and the properties of the document are
    stored in the Chunk with other important data.
    """

    @abstractmethod
    def split(self, input_document: InputDocument, parent_chunk: Chunk = None) -> List[Chunk]:
        pass

    @staticmethod
    @abstractmethod
    def name() -> str:
        pass
    