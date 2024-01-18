from rag4p.domain.chunk import Chunk
from rag4p.domain.relevant_chunk import RelevantChunk
from rag4p.retrieval.retriever import Retriever
from rag4p.store.internal_content_store import InternalContentStore


class InternalContentRetriever(Retriever):
    def __init__(self, internal_content_store: InternalContentStore):
        self.content_store = internal_content_store

    def find_relevant_chunks(self, question: str, max_results: int = 4) -> [RelevantChunk]:
        return self.content_store.find_relevant_chunks(question, max_results)

    def get_chunk(self, document_id: str, chunk_id: int) -> Chunk:
        return self.content_store.get_chunk(document_id + "_" + str(chunk_id))

    def loop_over_chunks(self):
        yield from self.content_store.loop_over_chunks()
