from abc import ABC, abstractmethod

from rag4p.rag.model.chunk import Chunk
from rag4p.rag.model.relevant_chunk import RelevantChunk


class Retriever(ABC):
    """
    This interface is used to retrieve relevant chunks for a given question. Next to the functions to retrieve the
    answer using the question or the vector representation of the question, it also provides a function to loop over
    all chunks. Finally, it contains a method to get a specific chunk.
    """

    @abstractmethod
    def find_relevant_chunks(self, question: str, max_results: int = 4) -> [RelevantChunk]:
        pass

    def get_chunk(self, document_id: str, chunk_id: str) -> Chunk:
        return self.get_chunk_by_id(document_id + "_" + chunk_id)

    @abstractmethod
    def get_chunk_by_id(self, chunk_id: str) -> Chunk:
        pass

    @abstractmethod
    def loop_over_chunks(self):
        pass
