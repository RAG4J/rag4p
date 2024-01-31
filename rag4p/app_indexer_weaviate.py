from rag4p.connectopenai.openai_embedder import OpenAIEmbedder
from rag4p.connectweaviate import chunk_collection, CLASS_NAME
from rag4p.connectweaviate.access_weaviate import AccessWeaviate
from rag4p.connectweaviate.weaviate_content_store import WeaviateContentStore
from rag4p.indexing.SentenceSplitter import SentenceSplitter
from rag4p.indexing.indexing_service import IndexingService
from rag4p.util.key_loader import KeyLoader
from rag4p.vasa_content_reader import VasaContentReader

if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()

    key_loader = KeyLoader()
    access_weaviate = AccessWeaviate(url=key_loader.get_weaviate_url(), access_key=key_loader.get_weaviate_api_key())
    access_weaviate.force_create_collection(collection_name=CLASS_NAME, properties=chunk_collection.properties)

    embedder = OpenAIEmbedder(api_key=key_loader.get_openai_api_key())
    content_store = WeaviateContentStore(weaviate_access=access_weaviate,embedder=embedder)
    splitter = SentenceSplitter()
    indexing_service = IndexingService(content_store=content_store)

    content_reader = VasaContentReader()
    indexing_service.index_documents(content_reader=content_reader, splitter=splitter)

    access_weaviate.close()

    # query = "Since when was the Vasa available for the public to visit?"
    # relevant_chunks = content_store.find_relevant_chunks(query=query, max_items=2)
    # print(f"Found {len(relevant_chunks)} relevant chunks for query: {query}")
    # for chunk in relevant_chunks:
    #     print(f"Document: {chunk.document_id}")
    #     print(f"Chunk id: {chunk.chunk_id}")
    #     print(f"Text: {chunk.chunk_text}")
