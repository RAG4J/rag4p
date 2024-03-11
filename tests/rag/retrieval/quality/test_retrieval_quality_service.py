import unittest
from unittest.mock import Mock

from rag4p.rag.model.relevant_chunk import RelevantChunk
from rag4p.rag.retrieval.quality.retrieval_quality_service import obtain_retrieval_quality, read_question_answers_from_file
from rag4p.rag.retrieval.quality.question_answer_record import QuestionAnswerRecord
from rag4p.rag.retrieval.retriever import Retriever


class TestRetrievalQualityService(unittest.TestCase):
    def setUp(self):
        self.retriever = Mock(spec=Retriever)
        self.question_answer_records = [
            QuestionAnswerRecord("doc1", 1, "text1", "question1"),
            QuestionAnswerRecord("doc2", 2, "text2", "question2"),
        ]

    def test_correct_retrieval_results(self):
        relevant_chunks = [RelevantChunk("doc1", 1, 1, "text1", {}, 0.7)]
        self.retriever.find_relevant_chunks = Mock(return_value=relevant_chunks)
        retrieval_quality = obtain_retrieval_quality(self.question_answer_records, self.retriever)
        self.assertEqual(0.5, retrieval_quality.precision())
        self.assertEqual(1, len(retrieval_quality.correct))
        self.assertEqual(1, len(retrieval_quality.incorrect))

    def test_read_question_answers_from_file_success(self):
        question_answers = read_question_answers_from_file("../tests/data/questions_answers.csv")
        self.assertEqual(len(question_answers), 1)

    def test_read_question_answers_from_file_failure(self):
        with self.assertRaises(IOError):
            read_question_answers_from_file("non_existent_file.csv")


if __name__ == '__main__':
    unittest.main()
