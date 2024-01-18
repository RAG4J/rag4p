from abc import ABC, abstractmethod

from rag4p.domain.chunk import Chunk
from rag4p.domain.relevant_chunk import RelevantChunk


class Retriever(ABC):

    @abstractmethod
    def find_relevant_chunks(self, question: str, max_results: int = 4) -> [RelevantChunk]:
        pass

    @abstractmethod
    def get_chunk(self, document_id: str, chunk_id: int) -> Chunk:
        pass

    @abstractmethod
    def loop_over_chunks(self):
        pass
