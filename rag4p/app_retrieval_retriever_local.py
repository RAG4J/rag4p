from rag4p.indexing.indexing_service import IndexingService
from rag4p.indexing.splitters.single_chunk_splitter import SingleChunkSplitter
from rag4p.integrations.ollama.access_ollama import AccessOllama
from rag4p.integrations.ollama.ollama_embedder import OllamaEmbedder
from rag4p.rag.store.local.internal_content_store import InternalContentStore
from rag4p.vasa_content_reader import VasaContentReader


if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()

    example_sentences = [
        "How many bolts were replaced?",
        "Since When could people visit the Vasa?",
        "Since when was the Vasa available for the public to visit?",
        "Who was responsible for building the Vasa ship?",
        "Where did the person responsible for building the Vasa ship come from?"
    ]

    splitter = SingleChunkSplitter()
    access_ollama = AccessOllama()
    embedder = OllamaEmbedder(access_ollama=access_ollama)
    content_store = InternalContentStore(embedder=embedder)

    indexing_service = IndexingService(content_store=content_store)
    indexing_service.index_documents(content_reader=VasaContentReader(), splitter=splitter)

    for example_sentence in example_sentences:
        relevant_chunks = content_store.find_relevant_chunks(example_sentence, max_results=2)
        print(f"Found {len(relevant_chunks)} relevant chunks for query: {example_sentence}")
        for chunk in relevant_chunks:
            print(f"Document: {chunk.document_id}")
            print(f"Chunk id: {chunk.chunk_id}")
            print(f"Text: {chunk.chunk_text}")
            print(f"Distance: {chunk.score:.3f}")