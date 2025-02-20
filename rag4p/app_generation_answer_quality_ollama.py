from rag4p.integrations.ollama.access_ollama import AccessOllama
from rag4p.integrations.ollama.quality.ollama_answer_quality_service import OllamaAnswerQualityService
from rag4p.logging_config import setup_logging
from rag4p.rag.tracker.rag_observer import RAGObserver

if __name__ == '__main__':
    from dotenv import load_dotenv

    load_dotenv()
    setup_logging()

    question = "Since when was the Vasa available for the public to visit?"
    answer = "The Vasa ship was available for the public to visit since Friday 16 February 1962."
    context = ("By Friday 16 February 1962, the ship is ready to be displayed to the general public at the "
               "newly-constructed Wasa Shipyard, where visitors can see Vasa while a team of conservators, "
               "carpenters and other technicians work to preserve the ship.")

    rag_observer = RAGObserver(question=question, answer=answer, context=context)

    answer_quality_service = OllamaAnswerQualityService(access_ollama=AccessOllama())

    quality = answer_quality_service.determine_quality_answer_related_to_question(rag_observer=rag_observer)
    print(f"Quality: {quality.quality}, Reason: {quality.reason}")

    quality = answer_quality_service.determine_quality_answer_from_context(rag_observer=rag_observer)
    print(f"Quality: {quality.quality}, Reason: {quality.reason}")