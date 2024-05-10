from openai import OpenAI

from rag4p.integrations.openai import MODEL_GPT4_TURBO
from rag4p.rag.generation.chat.chat_prompt import ChatPrompt
from rag4p.rag.generation.quality.answer_quality_service import AnswerQualityService
from rag4p.rag.tracker.rag_observer import RAGObserver


class OpenAIAnswerQualityService(AnswerQualityService):

    def __init__(self, openai_api_key: str, openai_model: str = MODEL_GPT4_TURBO):
        self.openai_client = OpenAI(
            api_key=openai_api_key,
        )
        self.openai_model = openai_model

    def obtain_answer_to_question_quality(self, chat_prompt: ChatPrompt, rag_observer: RAGObserver):
        response = self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system",
                 "content": chat_prompt.create_system_message(params={})},
                {"role": "user",
                 "content": chat_prompt.create_user_message(params={
                     "question": rag_observer.question,
                     "answer": rag_observer.answer})},
            ],
            stream=False,
        )

        return response.choices[0].message.content

    def obtain_answer_from_context_quality(self, chat_prompt: ChatPrompt, rag_observer: RAGObserver):
        response = self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system",
                 "content": chat_prompt.create_system_message(params={})},
                {"role": "user",
                 "content": chat_prompt.create_user_message(params={
                     "context": rag_observer.context,
                     "answer": rag_observer.answer})},
            ],
            stream=False,
        )
        return response.choices[0].message.content