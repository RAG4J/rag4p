from typing import List

from rag4p.integrations.weaviate import COLLECTION_NAME
from rag4p.integrations.weaviate.access_weaviate import AccessWeaviate
from rag4p.rag.model.chunk import Chunk
from rag4p.rag.store.content_store import ContentStore
from rag4p.rag.embedding.embedder import Embedder


class WeaviateContentStore(ContentStore):

    def __init__(self, weaviate_access: AccessWeaviate, embedder: Embedder, collection_name: str = COLLECTION_NAME):
        super().__init__({
            'name': 'weaviate-content-store',
            'embedder': embedder.identifier(),
            'collection_name': collection_name,
        })
        self.embedder = embedder
        self.weaviate_access = weaviate_access
        self.collection_name = collection_name

    def store(self, chunks: List[Chunk]):
        for chunk in chunks:
            properties = {
                "documentId": chunk.document_id,
                "chunkId": chunk.chunk_id,
                "text": chunk.chunk_text,
                "totalChunks": len(chunks),
            }

            for key, value in chunk.properties.items():
                properties[key] = value

            self.weaviate_access.add_document(
                collection_name=self.collection_name,
                properties=properties,
                vector=self.embedder.embed(chunk.chunk_text)
            )
