import json
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
        chat_prompt = ChatPrompt(system_message=self.quality_of_answer_to_question_system_prompt(),
                                 user_message=self.quality_of_answer_to_question_user_prompt())

        completion = self.obtain_answer_to_question_quality(chat_prompt, rag_observer)
        quality, reason = self.split_quality_and_reason(completion)
        return AnswerToQuestionQuality(quality, reason)

    @abstractmethod
    def obtain_answer_to_question_quality(self, chat_prompt: ChatPrompt, rag_observer: RAGObserver):
        pass

    def determine_quality_answer_from_context(self, rag_observer: RAGObserver) -> AnswerFromContextQuality:
        chat_prompt = ChatPrompt(system_message=self.quality_of_answer_from_context_system_prompt(),
                                 user_message=self.quality_of_answer_from_context_user_prompt())

        completion = self.obtain_answer_from_context_quality(chat_prompt, rag_observer)
        quality, reason = self.split_quality_and_reason(completion)
        return AnswerFromContextQuality(quality, reason)

    @abstractmethod
    def obtain_answer_from_context_quality(self, chat_prompt: ChatPrompt, rag_observer: RAGObserver):
        pass

    @staticmethod
    def split_quality_and_reason(quality_and_reason: str) -> (int, str):
        try:
            response = json.loads(quality_and_reason)
        except json.JSONDecodeError:
            print(f"Error: Could not decode response from LLM: {quality_and_reason}")
            return 0, "PROBLEM"
        return int(response["score"]), response["reason"]

    @staticmethod
    def quality_of_answer_to_question_system_prompt():
        return """You are a quality assistant verifying retrieval augmented generation systems. Your task is to 
        verify a generated answer against the proposed question. Give the answer a score between 1 and 5 and keep the 
        number as an integer. 5 means the answer contains the answer to the proposed question completely. 1 means 
        there is no match between the answer and the question at all. Keep the reason short as in maximum 2 sentences. 
        The question provided after 'question:'. The answer after 'answer:'. Write your answers in json format of: 
        {{"score": "score", "reason": "reason"}}
        An example: {{"score": 3,  "reason": "The answer is correct but some details are missing."}}"""


    @staticmethod
    def quality_of_answer_from_context_system_prompt():
        return """You are a quality assistant verifying retrieval augmented generation systems. Your task is to verify 
                a generated answer against the provided context. Give the answer a score between 1 and 5 and keep 
                the number as an integer. 5 means the answer contains only facts from the context. 1 means there is 
                not match between the answer and the provided context at all. If the answer contains exact phrases 
                from the context, the score should be lower as well. Keep the reason short as in maximum 2 sentences. 
                The answer provided after 'answer:'. The context after 'context:'. Write your answers in the json 
                format {{"score": "score", "reason": "reason"}}.  
                An example: {{"score": 3, "reason": "The answer is correct but some details are missing."}}"""

    @staticmethod
    def quality_of_answer_to_question_user_prompt():
        return "Question: {question}\nAnswer: {answer}\nResult:\n"

    @staticmethod
    def quality_of_answer_from_context_user_prompt():
        return "Answer: {answer}\nContext: {context}\nResult:\n"
