import os
from abc import ABC, abstractmethod

from openai import OpenAI

from rag4p.rag.generation.chat.chat_prompt import ChatPrompt
from rag4p.rag.generation.quality.answer_from_context_quality import AnswerFromContextQuality
from rag4p.rag.generation.quality.answer_quality import AnswerQuality
from rag4p.rag.generation.quality.answer_to_question_quality import AnswerToQuestionQuality
from rag4p.rag.tracker.rag_observer import RAGObserver


class AnswerQualityService(ABC):

    def determine_quality_of_answer(self, rag_observer: RAGObserver) -> AnswerQuality:
        answer_from_context = self.determine_quality_answer_from_context(rag_observer)
        answer_to_question = self.determine_quality_answer_related_to_question(rag_observer)

        return AnswerQuality(answer_to_question, answer_from_context)

    def determine_quality_answer_related_to_question(self, rag_observer: RAGObserver) -> AnswerToQuestionQuality:
        chat_prompt = ChatPrompt(system_message_filename="../data/quality/quality_of_answer_to_question_system.txt",
                                 user_message_filename="../data/quality/quality_of_answer_to_question_user.txt")

        completion = self.obtain_answer_to_question_quality(chat_prompt, rag_observer)
        quality, reason = self.split_quality_and_reason(completion)
        return AnswerToQuestionQuality(quality, reason)

    @abstractmethod
    def obtain_answer_to_question_quality(self, chat_prompt: ChatPrompt, rag_observer: RAGObserver):
        pass

    def determine_quality_answer_from_context(self, rag_observer: RAGObserver) -> AnswerFromContextQuality:
        chat_prompt = ChatPrompt(system_message_filename="../data/quality/quality_of_answer_from_context_system.txt",
                                 user_message_filename="../data/quality/quality_of_answer_from_context_user.txt")

        completion = self.obtain_answer_from_context_quality(chat_prompt, rag_observer)
        quality, reason = self.split_quality_and_reason(completion)
        return AnswerFromContextQuality(quality, reason)

    @abstractmethod
    def obtain_answer_from_context_quality(self, chat_prompt: ChatPrompt, rag_observer: RAGObserver):
        pass

    @staticmethod
    def split_quality_and_reason(quality_and_reason: str) -> (int, str):
        split = quality_and_reason.split("-", 1)
        quality = int(split[0].strip())
        reason = split[1].strip()
        return quality, reason