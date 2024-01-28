from __future__ import annotations

import csv
import os

from openai import OpenAI

from rag4p.connectopenai import DEFAULT_MODEL
from rag4p.generation.question_generator import QuestionGenerator
from rag4p.retrieval.retriever import Retriever


class QuestionGeneratorService:

    def __init__(self, retriever: Retriever, question_generator: QuestionGenerator):
        self.retriever = retriever
        self.question_generator = question_generator

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
        return self.question_generator.generate_question(context)