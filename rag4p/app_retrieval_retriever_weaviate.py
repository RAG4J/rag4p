from rag4p.integrations.openai.openai_embedder import OpenAIEmbedder
from rag4p.integrations.weaviate.access_weaviate import AccessWeaviate
from rag4p.integrations.weaviate.weaviate_retriever import WeaviateRetriever
from rag4p.logging_config import setup_logging
from rag4p.util.key_loader import KeyLoader

if __name__ == '__main__':
    from dotenv import load_dotenv

    load_dotenv()
    setup_logging()

    key_loader = KeyLoader()

    client = AccessWeaviate(url=key_loader.get_weaviate_url(), access_key=key_loader.get_weaviate_api_key())

    client.print_meta()

    embedder = OpenAIEmbedder(api_key=key_loader.get_openai_api_key())
    retriever = WeaviateRetriever(weaviate_access=client,
                                  embedder=embedder,
                                  additional_properties=["title", "speakers"],
                                  hybrid=True)

    for chunk in retriever.loop_over_chunks():
        print(f"Document: {chunk.document_id} - {chunk.chunk_id}")

    get_chunk = retriever.get_chunk(document_id="using-ai-to-save-time-and-sanity", chunk_id=0)
    print("--------------------------------------------------")
    print(f"Document: {get_chunk.document_id} - {get_chunk.chunk_id}")

    relevant_chunks = retriever.find_relevant_chunks("What are jettro and daniel talking about?", max_results=2)
    print(f"Found {len(relevant_chunks)} relevant chunks for query: What are jettro and daniel talking about?")

    for chunk in relevant_chunks:
        print(f"Document: {chunk.document_id}")
        print(f"Chunk id: {chunk.chunk_id}")
        print(f"Title: {chunk.properties['title']}")
        print(f"Speakers: {chunk.properties['speakers']}")
        print(f"Text: {chunk.chunk_text}")
        print(f"Score: {chunk.score}")
        print("--------------------------------------------------")

    client.close()
