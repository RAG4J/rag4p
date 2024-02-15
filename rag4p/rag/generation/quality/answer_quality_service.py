import os

from openai import OpenAI

from rag4p.rag.generation.chat.chat_prompt import ChatPrompt
from rag4p.rag.generation.quality.answer_from_context_quality import AnswerFromContextQuality
from rag4p.rag.generation.quality.answer_quality import AnswerQuality
from rag4p.rag.generation.quality.answer_to_question_quality import AnswerToQuestionQuality
from rag4p.rag.tracker.rag_observer import RAGObserver


class AnswerQualityService:
    def __init__(self, openai_api_key: str):
        self.openai_client = OpenAI(
            api_key=openai_api_key,
        )

    def determine_quality_of_answer(self, rag_observer: RAGObserver) -> AnswerQuality:
        answer_from_context = self.determine_quality_answer_from_context(rag_observer)
        answer_to_question = self.determine_quality_answer_related_to_question(rag_observer)

        return AnswerQuality(answer_to_question, answer_from_context)

    def determine_quality_answer_related_to_question(self, rag_observer: RAGObserver) -> AnswerToQuestionQuality:
        chat_prompt = ChatPrompt(system_message_filename="../data/quality/quality_of_answer_to_question_system.txt",
                                 user_message_filename="../data/quality/quality_of_answer_to_question_user.txt")

        completion = self.openai_client.chat.completions.create(
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
        quality, reason = self.split_quality_and_reason(completion.choices[0].message.content)
        return AnswerToQuestionQuality(quality, reason)

    def determine_quality_answer_from_context(self, rag_observer: RAGObserver) -> AnswerFromContextQuality:
        chat_prompt = ChatPrompt(system_message_filename="../data/quality/quality_of_answer_from_context_system.txt",
                                 user_message_filename="../data/quality/quality_of_answer_from_context_user.txt")

        completion = self.openai_client.chat.completions.create(
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
        quality, reason = self.split_quality_and_reason(completion.choices[0].message.content)
        return AnswerFromContextQuality(quality, reason)

    def split_quality_and_reason(self, quality_and_reason: str) -> (int, str):
        split = quality_and_reason.split("-", 1)
        quality = int(split[0].strip())
        reason = split[1].strip()
        return quality, reason