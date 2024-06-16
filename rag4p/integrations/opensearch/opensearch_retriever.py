from rag4p.integrations.opensearch import DEFAULT_INDEX
from rag4p.integrations.opensearch.opensearch_client import OpenSearchClient
from rag4p.rag.embedding.embedder import Embedder
from rag4p.rag.model.chunk import Chunk
from rag4p.rag.model.relevant_chunk import RelevantChunk
from rag4p.rag.retrieval.retriever import Retriever


class OpenSearchRetriever(Retriever):

    def __init__(self, opensearch_client: OpenSearchClient, embedder: Embedder,
                 additional_properties=None, hybrid: bool = False, index_name: str = DEFAULT_INDEX):
        if additional_properties is None:
            additional_properties = []

        self.opensearch_client = opensearch_client
        self.embedder = embedder
        self.index_name = index_name
        self.additional_properties = additional_properties
        self.hybrid = hybrid

    def find_relevant_chunks(self, question: str, max_results: int = 4) -> [RelevantChunk]:
        fields = ["chunk_text"]
        fields.extend(self.additional_properties)
        if not self.hybrid:
            query = {"match_all": {}}
        else:
            query = {"multi_match": {"fields": fields, "query": question}}
        query = {
            "_source": {
                "excludes": ["chunk_vector"]
            },
            "query": {
                "bool": {
                    "should": [
                        {
                            "script_score": {
                                "query": {"match_all": {}},
                                "script": {
                                    "source": "knn_score",
                                    "lang": "knn",
                                    "params": {
                                        "field": "chunk_vector",
                                        "query_value": self.embedder.embed(question),
                                        "space_type": "cosinesimil"
                                    }
                                }
                            }
                        },
                        query
                    ]
                }
            },
            "size": max_results
        }
        search_response = self.opensearch_client.search(body=query, index_name=self.index_name, size=max_results)
        relevant_chunks = []

        if search_response["hits"]["total"]["value"] == 0:
            return []

        for hit in search_response["hits"]["hits"]:
            properties = {}
            for key in self.additional_properties:
                properties[key] = hit["_source"][key]

            score = hit["_score"]
            relevant_chunks.append(RelevantChunk(
                document_id=hit["_source"]["document_id"],
                chunk_id=hit["_source"]["chunk_id"],
                text=hit["_source"]["chunk_text"],
                total_chunks=hit["_source"]["total_chunks"],
                properties=properties,
                score=score,
            ))
        return relevant_chunks

    def get_chunk_by_id(self, chunk_id: str) -> Chunk:
        chunk_response = self.opensearch_client.client().get(index=self.index_name, id=chunk_id)

        return Chunk(
            document_id=chunk_response["_source"]["document_id"],
            chunk_id=chunk_response["_source"]["chunk_id"],
            chunk_text=chunk_response["_source"]["chunk_text"],
            total_chunks=chunk_response["_source"]["total_chunks"],
            properties=chunk_response["_source"],
        )

    def loop_over_chunks(self):
        raw_client = self.opensearch_client.client()

        scroll_time = "2m"
        page_size = 1000

        response = raw_client.search(index=self.index_name, scroll=scroll_time, size=page_size)
        scroll_id = response['_scroll_id']
        hits = response['hits']['hits']

        chunks = []
        while len(hits) > 0:
            # Process the current batch of documents
            for hit in hits:
                properties = {}
                for key in self.additional_properties:
                    properties[key] = hit["_source"][key]

                chunks.append(Chunk(
                    document_id=hit["_source"]["document_id"],
                    chunk_id=hit["_source"]["chunk_id"],
                    chunk_text=hit["_source"]["chunk_text"],
                    total_chunks=hit["_source"]["total_chunks"],
                    properties=properties,
                ))

            # Fetch the next batch of documents
            response = raw_client.scroll(
                scroll_id=scroll_id,
                scroll=scroll_time
            )
            # Extract the scroll ID and the next batch of documents
            scroll_id = response['_scroll_id']
            hits = response['hits']['hits']

        # Clean up the scroll context when done
        raw_client.clear_scroll(scroll_id=scroll_id)

        return chunks
