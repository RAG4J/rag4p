import csv
import logging
import os

from rag4p.rag.retrieval.quality.question_answer_record import QuestionAnswerRecord
from rag4p.rag.retrieval.quality.retrieval_quality import RetrievalQuality
from rag4p.rag.retrieval.retriever import Retriever


rqs_logger = logging.getLogger(__name__)


def obtain_retrieval_quality(question_answer_records: [QuestionAnswerRecord], retriever: Retriever) -> RetrievalQuality:
    correct = []
    incorrect = []

    for item in question_answer_records:
        question = item.question
        document_id = item.document_id
        chunk_id = item.chunk_id

        # Retrieve the top 10 chunks for this question
        retrieved_chunks = retriever.find_relevant_chunks(question, max_results=1)[0]

        correct_answer = retrieved_chunks.document_id == document_id and retrieved_chunks.chunk_id == chunk_id
        if correct_answer:
            correct.append(item.document_id + " " + str(item.chunk_id))
        else:
            incorrect.append(item.document_id + " " + str(item.chunk_id))

        rqs_logger.info("Question: %s, Correct: %s", question, correct_answer)

    return RetrievalQuality(correct, incorrect)


def read_question_answers_from_file(file_name: str = "../data/questions_answers.csv"):
    directory = os.getcwd()
    file_path = os.path.join(directory, file_name)

    try:
        questions_answers = []
        with open(file_path, 'r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            next(reader)
            for row in reader:
                document_id = row[0]
                chunk_id = row[1]
                chunk_text = row[2]
                question = row[3]

                questions_answers.append(QuestionAnswerRecord(document_id, chunk_id, chunk_text, question))
        return questions_answers
    except IOError:
        rqs_logger.exception("Error while reading file with questions and answers in retrieval_quality_service.read_question_answers_from_file.")
        raise
