from rag4p.integrations.openai.openai_embedder import OpenAIEmbedder
from rag4p.integrations.opensearch.connection_builder import build_aws_search_service
from rag4p.integrations.opensearch.opensearch_client import OpenSearchClient
from rag4p.integrations.opensearch.opensearch_retriever import OpenSearchRetriever
from rag4p.integrations.weaviate.access_weaviate import AccessWeaviate
from rag4p.integrations.weaviate.weaviate_retriever import WeaviateRetriever
from rag4p.logging_config import setup_logging
from rag4p.util.key_loader import KeyLoader

if __name__ == '__main__':
    from dotenv import load_dotenv

    load_dotenv()
    setup_logging()
    key_loader = KeyLoader()

    opensearch_conn = build_aws_search_service(stack_name=key_loader.get_property("OPENSEARCH_STACK_NAME"),
                                               application_prefix=key_loader.get_property("OPENSEARCH_APP_PREFIX"))
    client = OpenSearchClient(client=opensearch_conn)

    embedder = OpenAIEmbedder(api_key=key_loader.get_openai_api_key())
    retriever = OpenSearchRetriever(opensearch_client=client,
                                    embedder=embedder,
                                    index_name="rag4p-vasa",
                                    additional_properties=["title", "timerange"],
                                    hybrid=False)

    for chunk in retriever.loop_over_chunks():
        print(f"Document: {chunk.document_id} - {chunk.chunk_id}")

    get_chunk = retriever.get_chunk(document_id="a-faithful-contract", chunk_id=4)
    print("--------------------------------------------------")
    print(f"Document: {get_chunk.document_id} - {get_chunk.chunk_id}")

    question = "shipwright"
    relevant_chunks = retriever.find_relevant_chunks(question=question, max_results=2)
    print(f"Found {len(relevant_chunks)} relevant chunks for query: {question}")

    for chunk in relevant_chunks:
        print(f"Document: {chunk.document_id}")
        print(f"Chunk id: {chunk.chunk_id}")
        print(f"Title: {chunk.properties['title']}")
        print(f"Time range: {chunk.properties['timerange']}")
        print(f"Text: {chunk.chunk_text}")
        print(f"Score: {chunk.score}")
        print("--------------------------------------------------")
