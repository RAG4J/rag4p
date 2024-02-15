from rag4p.rag.generation.answer_generator import AnswerGenerator
from rag4p.rag.tracker.rag_tracker import global_data


class ObservedAnswerGenerator(AnswerGenerator):
    def __init__(self, answer_generator: AnswerGenerator):
        self.answer_generator = answer_generator

    def generate_answer(self, question: str, context: str) -> str:
        answer = self.answer_generator.generate_answer(question, context)
        global_data["observer"].question = question
        global_data["observer"].context = context
        global_data["observer"].answer = answer
        return answer
