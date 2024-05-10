from rag4p.integrations.ollama import MODEL_PHI3
from rag4p.integrations.ollama.access_ollama import AccessOllama
from rag4p.rag.generation.chat.chat_prompt import ChatPrompt
from rag4p.rag.generation.quality.answer_quality_service import AnswerQualityService
from rag4p.rag.tracker.rag_observer import RAGObserver


class OllamaAnswerQualityService(AnswerQualityService):

    def __init__(self, access_ollama: AccessOllama):
        self.ollama = access_ollama

    def obtain_answer_from_context_quality(self, chat_prompt: ChatPrompt, rag_observer: RAGObserver):
        prompt_user = chat_prompt.create_user_message(params={
            "context": rag_observer.context,
            "answer": rag_observer.answer
        })
        prompt_system = chat_prompt.create_system_message(params={})
        prompt = prompt_system + prompt_user
        response = self.ollama.generate_answer(prompt=prompt, model=MODEL_PHI3)
        return response

    def obtain_answer_to_question_quality(self, chat_prompt: ChatPrompt, rag_observer: RAGObserver):
        prompt_user = chat_prompt.create_user_message(params={
            "question": rag_observer.question,
            "answer": rag_observer.answer
        })
        prompt_system = chat_prompt.create_system_message(params={})
        prompt = prompt_system + prompt_user
        response = self.ollama.generate_answer(prompt=prompt, model=MODEL_PHI3)
        return response

