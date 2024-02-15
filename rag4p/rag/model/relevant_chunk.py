from rag4p.rag.model.chunk import Chunk


class RelevantChunk (Chunk):
    score: float

    def __init__(self, document_id, chunk_id, total_chunks, text, properties, score):
        super().__init__(document_id, chunk_id, total_chunks, text, properties)
        self.score = score
