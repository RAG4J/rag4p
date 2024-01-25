import numpy as np

from rag4p.connectopenai.openai_answer_generator import OpenaiAnswerGenerator
from rag4p.connectopenai.openai_embedder import OpenAIEmbedder
from rag4p.connectweaviate.access_weaviate import AccessWeaviate
from rag4p.connectweaviate.weaviate_retriever import WeaviateRetriever
from rag4p.generation.observed_answer_generator import ObservedAnswerGenerator
from rag4p.indexing.SentenceSplitter import SentenceSplitter
from rag4p.indexing.indexing_service import IndexingService
from rag4p.onnxembedder.onnx_embedder import OnnxEmbedder
from rag4p.quality.answer_quality_service import AnswerQualityService
from rag4p.retrieval.topn_retrieval_strategy import TopNRetrievalStrategy
from rag4p.retrieval.window_retrieval_strategy import WindowRetrievalStrategy
from rag4p.store.internal_content_retriever import InternalContentRetriever
from rag4p.store.internal_content_store import InternalContentStore
from rag4p.tracker.rag_tracker import global_data
from rag4p.util.key_loader import KeyLoader
from rag4p.vasa_content_reader import VasaContentReader

if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()

    key_loader = KeyLoader()

    # Using the internal content store with the local embedder
    embedder = OnnxEmbedder()
    content_store = InternalContentStore(embedder=embedder)
    indexing_service = IndexingService(content_store=content_store)
    indexing_service.index_documents(content_reader=VasaContentReader(), splitter=SentenceSplitter())
    retriever = InternalContentRetriever(internal_content_store=content_store)

    # When using Weaviate, we do not need to index, that is done separately
    # Don't forget to close the client at the end of the script
    # embedder = OpenAIEmbedder(api_key=key_loader.get_openai_api_key())
    # weaviate_client = AccessWeaviate(url=key_loader.get_weaviate_url(), access_key=key_loader.get_weaviate_api_key())
    # retriever = WeaviateRetriever(weaviate_access=weaviate_client, embedder=embedder)

    openai_answer_generator = OpenaiAnswerGenerator(openai_api_key=key_loader.get_openai_api_key())
    answer_generator = ObservedAnswerGenerator(answer_generator=openai_answer_generator)

    example_sentences = [
        "How many bolts were replaced?",
        "Since When could people visit the Vasa?",
        "Since when was the Vasa available for the public to visit?",
        "Who was responsible for building the Vasa ship?",
        "Where did the person responsible for building the Vasa ship come from?"
    ]

    # strategy = TopNRetrievalStrategy(retriever=retriever)
    strategy = WindowRetrievalStrategy(retriever=retriever, window_size=1)
    answer_quality_service = AnswerQualityService(openai_api_key=key_loader.get_openai_api_key())

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
        print(f"Quality: {quality.quality}, Reason: {quality.reason}")

        quality = answer_quality_service.determine_quality_answer_from_context(rag_observer=rag_observer)
        answer_context_quality.append(quality.quality)
        print(f"Quality: {quality.quality}, Reason: {quality.reason}")

        rag_observer.reset()

    print("\n----------------------------------")
    print(f"Question Quality: {np.mean(answer_question_quality)}")
    print(f"Context Quality: {np.mean(answer_context_quality)}")

    # weaviate_client.close()
