from rag4p.indexing.indexing_service import IndexingService
from rag4p.indexing.splitters.sentence_splitter import SentenceSplitter
from rag4p.integrations.ollama.access_ollama import AccessOllama
from rag4p.integrations.ollama.ollama_embedder import OllamaEmbedder
from rag4p.logging_config import setup_logging
from rag4p.rag.retrieval.strategies.window_retrieval_strategy import WindowRetrievalStrategy
from rag4p.rag.store.local.internal_content_store import InternalContentStore
from rag4p.vasa_content_reader import VasaContentReader

if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    setup_logging()

    access_ollama = AccessOllama()
    embedder = OllamaEmbedder(access_ollama=access_ollama)
    content_store = InternalContentStore(embedder=embedder)
    indexing_service = IndexingService(content_store=content_store)
    indexing_service.index_documents(content_reader=VasaContentReader(), splitter=SentenceSplitter())

    example_sentences = [
        "How many bolts were replaced?",
        "Since When could people visit the Vasa?",
        "Since when was the Vasa available for the public to visit?",
        "Who was responsible for building the Vasa ship?",
        "Where did the person responsible for building the Vasa ship come from?"
    ]

    strategy = WindowRetrievalStrategy(retriever=content_store, window_size=1)

    print("\n\n----------------------------------")
    print(f"Retrieval Strategy {strategy.__class__.__name__}\n")
    for example_sentence in example_sentences:
        print("\n----------------------------------")
        retrieval_output = strategy.retrieve_max_results(question=example_sentence, max_results=2)

        print(f"Found {len(retrieval_output.items)} relevant chunks for query: {example_sentence}")
        for item in retrieval_output.items:
            print(f"Document: {item.document_id}, Chunk id: {item.chunk_id}, Text: {item.text}")

        print(f"Text: {retrieval_output.construct_context()}")
