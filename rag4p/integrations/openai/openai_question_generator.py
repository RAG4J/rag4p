from openai import OpenAI

from rag4p.rag.generation.question_generator import QuestionGenerator


class OpenAIQuestionGenerator(QuestionGenerator):

    def __init__(self, openai_api_key: str, openai_model: str):
        self.openai_client = OpenAI(
            api_key=openai_api_key,
        )
        self.openai_model = openai_model

    def generate_question(self, context: str) -> str:
        """
        Generates a question from a context.

        :param context: The context to generate a question from.
        :return: The generated question.
        """
        completion = self.openai_client.chat.completions.create(
            model=self.openai_model,
            messages=[
                {"role": "system",
                 "content": "You are a content writer reading a text and writing questions that are answered in that "
                            "text. Use the context as provided and nothing else to come up with one question. The "
                            "question should be a question that a person that does not know a lot about the context "
                            "could ask. Do not use names in your question or exact dates. Do not use the exact words "
                            "in the context. Each question must be one sentence only end always end with a '?' "
                            "character. The context is provided after 'context:'. The result should only contain the "
                            "generated question, nothing else."},
                {"role": "user",
                 "content": f"Context: {context}\nGenerated Question:"},
            ],
            stream=False,
        )
        return completion.choices[0].message.content
