from weaviate.classes import Filter
from weaviate.collections import Collection
import weaviate.classes as wvc

from rag4p.connectweaviate.access_weaviate import AccessWeaviate
from rag4p.domain.chunk import Chunk
from rag4p.domain.relevant_chunk import RelevantChunk
from rag4p.indexing.embedder import Embedder
from rag4p.retrieval.retriever import Retriever


class WeaviateRetriever(Retriever):

    def __init__(self, weaviate_access: AccessWeaviate, embedder: Embedder):
        self.weaviate_access = weaviate_access
        self.embedder = embedder

    def find_relevant_chunks(self, question: str, max_results: int = 4) -> [RelevantChunk]:
        vector = self.embedder.embed(question)
        result = self.__chunk_collection().query.near_vector(near_vector=vector,
                                                             limit=max_results,
                                                             return_metadata=wvc.query.MetadataQuery(distance=True))

        relevant_chunks = []
        for chunk in result.objects:
            relevant_chunks.append(RelevantChunk(
                document_id=chunk.properties["documentId"],
                chunk_id=chunk.properties["chunkId"],
                text=chunk.properties["text"],
                total_chunks=chunk.properties["totalChunks"],
                properties={},
                score=chunk.metadata.distance,
            ))
        return relevant_chunks

    def get_chunk(self, document_id: str, chunk_id: int) -> Chunk:
        filters = Filter.by_property("documentId").equal(document_id) & Filter.by_property("chunkId").equal(chunk_id)

        chunk = self.__chunk_collection().query.fetch_objects(limit=10, filters=filters)
        if len(chunk.objects) == 0:
            raise Exception(f"Chunk with documentId {document_id} and chunkId {chunk_id} not found")
        chunk = chunk.objects[0]

        return Chunk(
            document_id=chunk.properties["documentId"],
            chunk_id=chunk.properties["chunkId"],
            chunk_text=chunk.properties["text"],
            total_chunks=chunk.properties["totalChunks"],
            properties={},
        )

    def loop_over_chunks(self):
        for chunk in self.__chunk_collection().iterator():
            yield from self.__extract_chunk(chunk)

    def __extract_chunk(self, chunk):
        yield Chunk(
            document_id=chunk.properties["documentId"],
            chunk_id=chunk.properties["chunkId"],
            chunk_text=chunk.properties["text"],
            total_chunks=chunk.properties["totalChunks"],
            properties={},
        )

    def __chunk_collection(self) -> Collection:
        return self.weaviate_access.client.collections.get("Chunk")