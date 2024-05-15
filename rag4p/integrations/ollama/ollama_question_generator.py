from rag4p.integrations.ollama import DEFAULT_MODEL
from rag4p.integrations.ollama.access_ollama import AccessOllama
from rag4p.integrations.ollama.ollama_answer_generator import OllamaAnswerGenerator
from rag4p.rag.generation.question_generator import QuestionGenerator


class OllamaQuestionGenerator(QuestionGenerator):

    def __init__(self, access_ollama: AccessOllama, model: str = DEFAULT_MODEL):
        self.ollama = access_ollama
        self.model = model

    def generate_question(self, context: str) -> str:
        """
        Generates a question from a context.

        :param context: The context to generate a question from.
        :return: The generated question.
        """
        prompt = f"""
        You are a content writer reading a text and writing questions that are answered in that text. Use the context as
        provided and nothing else to come up with one question. The question should be a question that a person that
        does not know a lot about the context could ask. Do not use names in your question or exact dates. Do not use
        the exact words in the context. Each question must be one sentence only end always end with a '?' character. The
        context is provided after 'context:'. The result should only contain the generated question, nothing else.
        context: {context}
        generated question:
        """
        return self.ollama.generate_answer(prompt=prompt, model=self.model)
