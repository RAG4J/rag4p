from rag4p.integrations.bedrock import MODEL_TITAN_EXPRESS
from rag4p.integrations.bedrock.access_bedrock import AccessBedrock
from rag4p.rag.generation.chat.chat_prompt import ChatPrompt
from rag4p.rag.generation.quality.answer_quality_service import AnswerQualityService
from rag4p.rag.tracker.rag_observer import RAGObserver


class BedrockAnswerQualityService(AnswerQualityService):

    def __init__(self, access_bedrock: AccessBedrock):
        self.bedrock = access_bedrock

    def obtain_answer_to_question_quality(self, chat_prompt: ChatPrompt, rag_observer: RAGObserver):
        prompt_user = chat_prompt.create_user_message(params={
            "context": rag_observer.context,
            "answer": rag_observer.answer
        })
        prompt_system = chat_prompt.create_system_message(params={})
        prompt = prompt_system + prompt_user
        return self.bedrock.generate_answer(prompt=prompt, model=MODEL_TITAN_EXPRESS)

    def obtain_answer_from_context_quality(self, chat_prompt: ChatPrompt, rag_observer: RAGObserver):
        prompt_user = chat_prompt.create_user_message(params={
            "context": rag_observer.context,
            "answer": rag_observer.answer
        })
        prompt_system = chat_prompt.create_system_message(params={})
        prompt = prompt_system + prompt_user
        return self.bedrock.generate_answer(prompt=prompt, model=MODEL_TITAN_EXPRESS)
