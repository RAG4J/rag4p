from rag4p.integrations.bedrock import DEFAULT_MODEL
from rag4p.integrations.bedrock.access_bedrock import AccessBedrock
from rag4p.rag.generation.answer_generator import AnswerGenerator


class BedrockAnswerGenerator(AnswerGenerator):
    def __init__(self, access_bedrock: AccessBedrock, model: str = DEFAULT_MODEL):
        self.access_bedrock = access_bedrock
        self.model = model

    def generate_answer(self, question: str, context: str) -> str:
        prompt = f"""
        You are an assistant answering questions using the context provided. If the context does not contain the
        answer, you should tell you cannot answer using the context.
        question: {question}
        context: {context}
        answer:
        """
        return self.access_bedrock.generate_answer(prompt=prompt, model=self.model)

