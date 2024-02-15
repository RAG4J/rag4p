from abc import ABC, abstractmethod


class AnswerGenerator(ABC):
    """
    Abstract class for generating answers to questions provided with a context.
    """
    @abstractmethod
    def generate_answer(self, question: str, context: str) -> str:
        pass
