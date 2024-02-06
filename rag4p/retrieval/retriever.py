from abc import ABC, abstractmethod

from rag4p.domain.chunk import Chunk
from rag4p.domain.relevant_chunk import RelevantChunk


class Retriever(ABC):
    """
    This interface is used to retrieve relevant chunks for a given question. Next to the functions to retrieve the
    answer using the question or the vector representation of the question, it also provides a function to loop over
    all chunks. Finally, it contains a method to get a specific chunk.
    """

    @abstractmethod
    def find_relevant_chunks(self, question: str, max_results: int = 4) -> [RelevantChunk]:
        pass

    @abstractmethod
    def get_chunk(self, document_id: str, chunk_id: int) -> Chunk:
        pass

    @abstractmethod
    def loop_over_chunks(self):
        pass
