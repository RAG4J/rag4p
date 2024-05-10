from rag4p.integrations.ollama import DEFAULT_MODEL
from rag4p.integrations.ollama.access_ollama import AccessOllama
from rag4p.rag.generation.answer_generator import AnswerGenerator


class OllamaAnswerGenerator(AnswerGenerator):

    def __init__(self, access_ollama: AccessOllama, model: str = DEFAULT_MODEL):
        self.ollama = access_ollama
        self.model = model

    def generate_answer(self, question: str, context: str) -> str:
        prompt = f"""
        You are an assistant answering questions using the context provided. If the context does not contain the
        answer, you should tell you cannot answer using the context.
        question: {question}
        context: {context}
        answer:
        """
        return self.ollama.generate_answer(prompt=prompt, model=self.model)


if __name__ == "__main__":
    ollama = AccessOllama()
    q_a = OllamaAnswerGenerator(ollama)

    question = "Who are the presenters of the workshop about LLMs?"
    context = ("At TeqNation Jettro and Daniel present the workshop where you can learn about the building blocks of "
               "Large Language Models.")

    answer = q_a.generate_answer(question, context)

    print(answer)
