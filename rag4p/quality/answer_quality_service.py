from openai import OpenAI

from rag4p.connectopenai import DEFAULT_MODEL
from rag4p.quality.answer_from_context_quality import AnswerFromContextQuality
from rag4p.quality.answer_quality import AnswerQuality
from rag4p.quality.answer_to_question_quality import AnswerToQuestionQuality
from rag4p.tracker.rag_observer import RAGObserver


class AnswerQualityService:
    def __init__(self, openai_api_key: str, openai_model: str = DEFAULT_MODEL):
        self.openai_client = OpenAI(
            api_key=openai_api_key,
        )

    def determine_quality_of_answer(self, rag_observer: RAGObserver) -> AnswerQuality:
        answer_from_context = self.determine_quality_answer_from_context(rag_observer)
        answer_to_question = self.determine_quality_answer_related_to_question(rag_observer)

        return AnswerQuality(answer_to_question, answer_from_context)

    def determine_quality_answer_related_to_question(self, rag_observer: RAGObserver) -> AnswerToQuestionQuality:
        completion = self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system",
                 "content": "You are a quality assistant verifying retrieval augmented generation systems. Your task "
                            "is to verify a generated answer against the proposed question. Give the answer a score "
                            "between 1 and 5 and keep the number as an integer. 5 means the answer contains the "
                            "answer to the proposed question completely. 1 means there is not match between the "
                            "answer and the question at all. The question provided after 'question:'. The answer "
                            "after 'answer:'. Write your answers in the format of score - reason. Keep the reason "
                            "short as in maximum 2 sentences. An example: 3 - The answer is correct but some details "
                            "are missing."},
                {"role": "user",
                 "content": f"Question: {rag_observer.question}\nAnswer: {rag_observer.answer}\nResult:"},
            ],
            stream=False,
        )
        quality, reason = self.split_quality_and_reason(completion.choices[0].message.content)
        return AnswerToQuestionQuality(quality, reason)

    def determine_quality_answer_from_context(self, rag_observer: RAGObserver) -> AnswerFromContextQuality:
        completion = self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system",
                 "content": "You are a quality assistant verifying retrieval augmented generation systems. Your task "
                            "is to verify a generated answer against the provided context. Give the answer a score "
                            "between 1 and 5 and keep the number as an integer. 5 means the answer contains only "
                            "facts from the context. 1 means there is not match between the answer and the provided "
                            "context at all. If the answer contains exact phrases from the context, the score should "
                            "be lower as well. The answer provided after 'answer:'. The context after 'context:'. "
                            "Write your answers in the format of score - reason. Keep the reason short as in maximum "
                            "2 sentences. An example: 3 - The answer is correct but some details are missing."},
                {"role": "user",
                 "content": f"Answer: {rag_observer.answer}\nContext: {rag_observer.context}\nResult:"},
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