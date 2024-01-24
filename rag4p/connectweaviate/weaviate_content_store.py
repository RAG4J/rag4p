from typing import List

from rag4p.connectweaviate.access_weaviate import AccessWeaviate
from rag4p.domain.chunk import Chunk
from rag4p.indexing.content_store import ContentStore
from rag4p.indexing.embedder import Embedder


class WeaviateContentStore(ContentStore):

    def __init__(self, weaviate_access: AccessWeaviate, embedder: Embedder):
        super().__init__(embedder)
        self.weaviate_access = weaviate_access

    def store(self, chunks: List[Chunk]):
        for chunk in chunks:
            properties = {
                "documentId": chunk.document_id,
                "chunkId": chunk.chunk_id,
                "text": chunk.chunk_text,
                "totalChunks": len(chunks)
            }

            self.weaviate_access.add_document(
                collection_name="Chunk",
                properties=properties,
                vector=self.embedder.embed(chunk.chunk_text)
            )
