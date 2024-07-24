import json

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
        answer, you should tell you cannot answer using the context. Use the json format for the answer.
        {{"answer": "answer"}}
        question: {question}
        context: {context}
        answer:
        """
        answer = self.ollama.generate_answer(prompt=prompt, model=self.model)
        json_answer = json.loads(answer)
        return json_answer["answer"]
