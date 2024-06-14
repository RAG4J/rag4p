from typing import List

from rag4p.integrations.opensearch import DEFAULT_INDEX
from rag4p.integrations.opensearch.opensearch_client import OpenSearchClient
from rag4p.rag.embedding.embedder import Embedder
from rag4p.rag.model.chunk import Chunk
from rag4p.rag.store.content_store import ContentStore


class OpenSearchContentStore(ContentStore):

    def __init__(self, opensearch_client: OpenSearchClient, embedder: Embedder, index_name: str = DEFAULT_INDEX):
        super().__init__({
            'name': 'opensearch-content-store',
            'embedder': embedder.identifier(),
            'index_name': index_name,
        })

        self.embedder = embedder
        self.index_name = index_name
        self.client = opensearch_client
        self.client.ping()

    def store(self, chunks: List[Chunk]):
        for chunk in chunks:
            properties = {
                "document_id": chunk.document_id,
                "chunk_id": chunk.chunk_id,
                "chunk_text": chunk.chunk_text,
                "total_chunks": len(chunks),
                "chunk_vector": self.embedder.embed(chunk.chunk_text),
            }

            for key, value in chunk.properties.items():
                properties[key] = value

            self.client.index_document(
                id=chunk.get_id(),
                document=properties,
                index_name=self.index_name,
            )
