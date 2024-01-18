import csv
import os

from openai import OpenAI

from rag4p.retrieval.retriever import Retriever


class QuestionGeneratorService:

    def __init__(self, openai_key: str, retriever: Retriever):
        self.openai_client = OpenAI(
            api_key=openai_key,
        )
        self.retriever = retriever

    def generate_question_answer_pairs(self, file_name: str):
        directory = os.getcwd()  # get current working directory
        file_path = os.path.join(directory, "../data", file_name)
        print(file_path)
        try:
            with open(file_path, 'w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(["document", "chunk", "text", "question"])
                for chunk in self.retriever.loop_over_chunks():
                    # Use LLM to generate a question for this chunk
                    question = self.generate_question(chunk.chunk_text)

                    # Write the question and answer to a file
                    document_id = chunk.document_id
                    chunk_id = str(chunk.chunk_id)
                    writer.writerow([document_id, chunk_id, chunk.chunk_text, question])
                    print(f"Generated question: {question}")
        except IOError as e:
            print("An error occurred while writing to the file.", e)
            raise

    def generate_question(self, context: str) -> str:
        """
        Generates a question from a context.

        :param context: The context to generate a question from.
        :return: The generated question.
        """
        completion = self.openai_client.chat.completions.create(
            model="gpt-4",
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
