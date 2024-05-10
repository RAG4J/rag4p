import numpy as np

from rag4p.indexing.indexing_service import IndexingService
from rag4p.indexing.splitters.sentence_splitter import SentenceSplitter
from rag4p.integrations.ollama.access_ollama import AccessOllama
from rag4p.integrations.ollama.ollama_answer_generator import OllamaAnswerGenerator
from rag4p.integrations.ollama.ollama_embedder import OllamaEmbedder
from rag4p.integrations.openai.quality.openai_answer_quality_service import OpenAIAnswerQualityService
from rag4p.rag.generation.observed_answer_generator import ObservedAnswerGenerator
from rag4p.rag.retrieval.strategies.window_retrieval_strategy import WindowRetrievalStrategy
from rag4p.rag.store.local.internal_content_store import InternalContentStore
from rag4p.rag.tracker.rag_tracker import global_data
from rag4p.util.key_loader import KeyLoader
from rag4p.vasa_content_reader import VasaContentReader

if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()

    key_loader = KeyLoader()

    # Using the internal content store with the local embedder
    access_ollama = AccessOllama()
    embedder = OllamaEmbedder(access_ollama=access_ollama)
    content_store = InternalContentStore(embedder=embedder)
    indexing_service = IndexingService(content_store=content_store)
    indexing_service.index_documents(content_reader=VasaContentReader(), splitter=SentenceSplitter())

    ollama_answer_generator = OllamaAnswerGenerator(access_ollama=access_ollama)
    answer_generator = ObservedAnswerGenerator(answer_generator=ollama_answer_generator)

    example_sentences = [
        "How many bolts were replaced?",
        "Since When could people visit the Vasa?",
        "Since when was the Vasa available for the public to visit?",
        "Who was responsible for building the Vasa ship?",
        "Where did the person responsible for building the Vasa ship come from?"
    ]

    # strategy = TopNRetrievalStrategy(retriever=retriever)
    strategy = WindowRetrievalStrategy(retriever=content_store, window_size=1)
    answer_quality_service = OpenAIAnswerQualityService(openai_api_key=key_loader.get_openai_api_key())

    answer_question_quality = []
    answer_context_quality = []
    for example_sentence in example_sentences:
        print("\n----------------------------------")
        retrieval_output = strategy.retrieve_max_results(question=example_sentence, max_results=1)

        # Generate answer using the obtained context
        answer = answer_generator.generate_answer(question=example_sentence,
                                                  context=retrieval_output.construct_context())

        print(f"Question: {example_sentence}")
        print(f"Answer: {answer}")
        print(f"Context: {retrieval_output.construct_context()}")

        rag_observer = global_data["observer"]

        quality = answer_quality_service.determine_quality_answer_related_to_question(rag_observer=rag_observer)
        answer_question_quality.append(quality.quality)
        print(f"Answer to Question Quality: {quality.quality}, Reason: {quality.reason}")

        quality = answer_quality_service.determine_quality_answer_from_context(rag_observer=rag_observer)
        answer_context_quality.append(quality.quality)
        print(f"Answer from Context Quality: {quality.quality}, Reason: {quality.reason}")

        rag_observer.reset()

    print("\n----------------------------------")
    print(f"Question Quality: {np.mean(answer_question_quality)}")
    print(f"Context Quality: {np.mean(answer_context_quality)}")
