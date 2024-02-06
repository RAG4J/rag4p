from weaviate.collections import Collection
import weaviate.classes as wvc
from weaviate.collections.classes.filters import Filter

from rag4p.connectweaviate import CLASS_NAME
from rag4p.connectweaviate.access_weaviate import AccessWeaviate
from rag4p.domain.chunk import Chunk
from rag4p.domain.relevant_chunk import RelevantChunk
from rag4p.indexing.embedder import Embedder
from rag4p.retrieval.retriever import Retriever


class WeaviateRetriever(Retriever):

    def __init__(self, weaviate_access: AccessWeaviate, embedder: Embedder, additional_properties=None):
        if additional_properties is None:
            additional_properties = []

        self.weaviate_access = weaviate_access
        self.embedder = embedder
        self.additional_properties = additional_properties

    def find_relevant_chunks(self, question: str, max_results: int = 4) -> [RelevantChunk]:
        vector = self.embedder.embed(question)
        result = self.__chunk_collection().query.near_vector(near_vector=vector,
                                                             limit=max_results,
                                                             return_metadata=wvc.query.MetadataQuery(distance=True))

        relevant_chunks = []
        for chunk in result.objects:
            properties = {}
            for key in self.additional_properties:
                properties[key] = chunk.properties[key]

            relevant_chunks.append(RelevantChunk(
                document_id=chunk.properties["documentId"],
                chunk_id=chunk.properties["chunkId"],
                text=chunk.properties["text"],
                total_chunks=chunk.properties["totalChunks"],
                properties=properties,
                score=chunk.metadata.distance,
            ))
        return relevant_chunks

    def get_chunk(self, document_id: str, chunk_id: int) -> Chunk:
        filters = Filter.by_property("documentId").equal(document_id) & Filter.by_property("chunkId").equal(chunk_id)

        chunk = self.__chunk_collection().query.fetch_objects(limit=10, filters=filters)
        if len(chunk.objects) == 0:
            raise Exception(f"Chunk with documentId {document_id} and chunkId {chunk_id} not found")
        chunk = chunk.objects[0]

        properties = {}
        for key in self.additional_properties:
            properties[key] = chunk.properties[key]

        return Chunk(
            document_id=chunk.properties["documentId"],
            chunk_id=chunk.properties["chunkId"],
            chunk_text=chunk.properties["text"],
            total_chunks=chunk.properties["totalChunks"],
            properties=properties,
        )

    def loop_over_chunks(self):
        for chunk in self.__chunk_collection().iterator():
            yield from self.__extract_chunk(chunk)

    def __extract_chunk(self, chunk):
        properties = {}
        for key in self.additional_properties:
            properties[key] = chunk.properties[key]

        yield Chunk(
            document_id=chunk.properties["documentId"],
            chunk_id=chunk.properties["chunkId"],
            chunk_text=chunk.properties["text"],
            total_chunks=chunk.properties["totalChunks"],
            properties=properties
        )

    def __chunk_collection(self) -> Collection:
        return self.weaviate_access.client.collections.get(CLASS_NAME)