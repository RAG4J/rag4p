from rag4p.indexing.indexing_service import IndexingService
from rag4p.indexing.splitters.sentence_splitter import SentenceSplitter
from rag4p.integrations.ollama.access_ollama import AccessOllama
from rag4p.integrations.ollama.ollama_embedder import OllamaEmbedder
from rag4p.rag.store.local.internal_content_store import InternalContentStore
from rag4p.util.key_loader import KeyLoader
from rag4p.vasa_content_reader import VasaContentReader

if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()

    key_loader = KeyLoader()
    access_ollama = AccessOllama()
    embedder = OllamaEmbedder(access_ollama=access_ollama)
    content_reader = VasaContentReader()
    content_store = InternalContentStore(embedder=embedder)
    splitter = SentenceSplitter()

    indexing_service = IndexingService(content_store=content_store)
    response = indexing_service.index_documents(content_reader=content_reader, splitter=splitter)

    print(response)

    query = "Since when was the Vasa available for the public to visit?"
    relevant_chunks = content_store.find_relevant_chunks(query=query, max_results=2)
    print(f"Found {len(relevant_chunks)} relevant chunks for query: {query}")
    for chunk in relevant_chunks:
        print(f"Document: {chunk.document_id}")
        print(f"Chunk id: {chunk.chunk_id}")
        print(f"Text: {chunk.chunk_text}")
