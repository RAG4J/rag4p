import csv
import os

from rag4p.rag.retrieval.quality.question_answer_record import QuestionAnswerRecord
from rag4p.rag.retrieval.quality.retrieval_quality import RetrievalQuality
from rag4p.rag.retrieval.retriever import Retriever


def obtain_retrieval_quality(question_answer_records: [QuestionAnswerRecord], retriever: Retriever) -> RetrievalQuality:
    correct = []
    incorrect = []

    for item in question_answer_records:
        question = item.question
        document_id = item.document_id
        chunk_id = item.chunk_id

        # Retrieve the top 10 chunks for this question
        retrieved_chunks = retriever.find_relevant_chunks(question, max_results=1)[0]

        if retrieved_chunks.document_id == document_id and retrieved_chunks.chunk_id == chunk_id:
            correct.append(item.document_id + " " + str(item.chunk_id))
        else:
            incorrect.append(item.document_id + " " + str(item.chunk_id))

    return RetrievalQuality(correct, incorrect)


def read_question_answers_from_file(file_name: str = "questions_answers.csv"):
    directory = os.getcwd()
    file_path = os.path.join(directory, "data", file_name)

    try:
        questions_answers = []
        with open(file_path, 'r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                document_id = row[0]
                chunk_id = int(row[1])
                chunk_text = row[2]
                question = row[3]

                questions_answers.append(QuestionAnswerRecord(document_id, chunk_id, chunk_text, question))
        return questions_answers
    except IOError as e:
        print("An error occurred while reading from the file.", e)
        raise
