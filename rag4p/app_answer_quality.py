import os

from rag4p.quality.answer_quality_service import AnswerQualityService
from rag4p.tracker.rag_observer import RAGObserver
from rag4p.util.key_loader import KeyLoader

if __name__ == '__main__':
    from dotenv import load_dotenv

    load_dotenv()

    question = "Since when was the Vasa available for the public to visit?"
    answer = "The Vasa ship was available for the public to visit since Friday 16 February 1962."
    context = ("By Friday 16 February 1962, the ship is ready to be displayed to the general public at the "
               "newly-constructed Wasa Shipyard, where visitors can see Vasa while a team of conservators, "
               "carpenters and other technicians work to preserve the ship.")

    rag_observer = RAGObserver(question=question, answer=answer, context=context)

    key_loader = KeyLoader()
    answer_quality_service = AnswerQualityService(openai_api_key=key_loader.get_openai_api_key())

    quality = answer_quality_service.determine_quality_answer_related_to_question(rag_observer=rag_observer)
    print(f"Quality: {quality.quality}, Reason: {quality.reason}")

    quality = answer_quality_service.determine_quality_answer_from_context(rag_observer=rag_observer)
    print(f"Quality: {quality.quality}, Reason: {quality.reason}")