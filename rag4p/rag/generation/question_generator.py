from abc import ABC, abstractmethod


class QuestionGenerator(ABC):
    @abstractmethod
    def generate_question(self, context: str) -> str:
        pass
