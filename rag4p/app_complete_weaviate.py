import numpy as np

from rag4p.integrations.openai.openai_answer_generator import OpenaiAnswerGenerator
from rag4p.integrations.openai.openai_embedder import OpenAIEmbedder
from rag4p.integrations.weaviate.access_weaviate import AccessWeaviate
from rag4p.integrations.weaviate.weaviate_retriever import WeaviateRetriever
from rag4p.rag.generation.observed_answer_generator import ObservedAnswerGenerator
from rag4p.rag.generation.quality.answer_quality_service import AnswerQualityService
from rag4p.rag.retrieval.strategies.window_retrieval_strategy import WindowRetrievalStrategy
from rag4p.rag.tracker.rag_tracker import global_data
from rag4p.util.key_loader import KeyLoader

if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    key_loader = KeyLoader()

    embedder = OpenAIEmbedder(api_key=key_loader.get_openai_api_key())
    weaviate_client = AccessWeaviate(url=key_loader.get_weaviate_url(), access_key=key_loader.get_weaviate_api_key())
    retriever = WeaviateRetriever(weaviate_access=weaviate_client, embedder=embedder, hybrid=False)

    openai_answer_generator = OpenaiAnswerGenerator(openai_api_key=key_loader.get_openai_api_key())
    answer_generator = ObservedAnswerGenerator(answer_generator=openai_answer_generator)

    example_sentences = [
        "How many bolts were replaced?",
        "Since When could people visit the Vasa?",
        "Since when was the Vasa available for the public to visit?",
        "Who was responsible for building the Vasa ship?",
        "Where did the person responsible for building the Vasa ship come from?"
    ]

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
        print(f"Answer to Question Quality: {quality.quality}, Reason: {quality.reason}")

        quality = answer_quality_service.determine_quality_answer_from_context(rag_observer=rag_observer)
        answer_context_quality.append(quality.quality)
        print(f"Answer from Context Quality: {quality.quality}, Reason: {quality.reason}")

        rag_observer.reset()

    print("\n----------------------------------")
    print(f"Question Quality: {np.mean(answer_question_quality)}")
    print(f"Context Quality: {np.mean(answer_context_quality)}")

    weaviate_client.close()
