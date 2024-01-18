from openai import OpenAI

from rag4p.generation.answer_generator import AnswerGenerator


class OpenaiAnswerGenerator(AnswerGenerator):
    def __init__(self, openai_api_key: str):
        self.openai_client = OpenAI(
            api_key=openai_api_key,
        )

    def generate_answer(self, question: str, context: str) -> str:
        completion = self.openai_client.chat.completions.create(
            model="gpt-4",
            messages=[
                {"role": "system", "content": "You are the tour guide for the Vasa Museum.You task is to answer "
                                              "question about the Vasa ship. Limit your answer to the context as "
                                              "provided. Do not use your own knowledge. The question is provided "
                                              "after 'question:'. The context after 'context:'."},
                {"role": "user", "content": f"Context: {context}\nQuestion: {question}\nAnswer:"},
            ],
            stream=False,
        )

        # print(completion.model_dump_json(indent=2))
        return completion.choices[0].message.content
