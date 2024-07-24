from rag4p.rag.model.chunk import Chunk
from rag4p.rag.model.relevant_chunk import RelevantChunk
from rag4p.rag.retrieval.retriever import Retriever
from rag4p.rag.tracker.rag_tracker import global_data


class ObservedRetriever(Retriever):
    def __init__(self, retriever: Retriever):
        self.retriever = retriever

    def find_relevant_chunks(self, question: str, max_results: int = 4) -> [RelevantChunk]:
        relevant_chunks = self.retriever.find_relevant_chunks(question, max_results)
        for relevant_chunk in relevant_chunks:
            global_data["observer"].add_relevant_chunk(relevant_chunk.get_id(), relevant_chunk.text)
        return relevant_chunks

    def get_chunk(self, document_id: str, chunk_id: str) -> Chunk:
        chunk = self.retriever.get_chunk(document_id, chunk_id)
        return chunk

    def loop_over_chunks(self):
        yield from self.retriever.loop_over_chunks()