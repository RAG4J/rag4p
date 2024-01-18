from rag4p.quality.answer_from_context_quality import AnswerFromContextQuality
from rag4p.quality.answer_to_question_quality import AnswerToQuestionQuality


class AnswerQuality:
    answer_to_question: AnswerToQuestionQuality
    answer_from_context: AnswerFromContextQuality

    def __init__(self, answer_to_question: AnswerToQuestionQuality, answer_from_context: AnswerFromContextQuality):
        self.answer_to_question = answer_to_question
        self.answer_from_context = answer_from_context
