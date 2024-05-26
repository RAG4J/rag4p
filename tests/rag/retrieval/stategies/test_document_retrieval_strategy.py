import unittest
from unittest.mock import Mock
from rag4p.rag.retrieval.strategies.document_retrieval_strategy import DocumentRetrievalStrategy
from rag4p.rag.retrieval.retriever import Retriever
from rag4p.rag.model.relevant_chunk import RelevantChunk, Chunk


class TestDocumentRetrievalStrategy(unittest.TestCase):
    def setUp(self):
        self.retriever = Mock(spec=Retriever)

        def mock_get_chunk(document_id, chunk_id):
            if document_id == 'doc1' and chunk_id == 0:
                return Chunk('doc1', 0, 3, "This is the text for chunk 1 of 3", {'prop1': 'value1'})
            elif document_id == 'doc1' and chunk_id == 1:
                return Chunk('doc1', 1, 3, "This is the text for chunk 2 of 3", {'prop1': 'value1'})
            elif document_id == 'doc1' and chunk_id == 2:
                return Chunk('doc1', 2, 3, "This is the text for chunk 3 of 3", {'prop1': 'value1'})
            elif document_id == 'doc2' and chunk_id == 0:
                return Chunk('doc2', 0, 2, "This is the text for chunk 1 of 2", {'prop1': 'value2'})
            elif document_id == 'doc2' and chunk_id == 1:
                return Chunk('doc1', 1, 2, "This is the text for chunk 2 of 2", {'prop1': 'value2'})
            else:
                return None

        # Set the side_effect attribute of the get_chunk method to the mock_get_chunk function
        self.retriever.get_chunk.side_effect = mock_get_chunk

    def test_retrieve_max_results_returns_unique_documents(self):
        # Mock the retriever's find_relevant_chunks method to return chunks with duplicate document_ids
        self.retriever.find_relevant_chunks.return_value = [
            RelevantChunk('doc1', 0, 3, "This is the text for chunk 1 of 3", {'prop1': 'value1'}, 0.8),
            RelevantChunk('doc1', 2, 3, "This is the text for chunk 3 of 3", {'prop1': 'value1'}, 0.7),
            RelevantChunk('doc2', 1, 2, "This is the text for chunk 2 of 2", {'prop1': 'value2'}, 0.6)
        ]
        strategy = DocumentRetrievalStrategy(self.retriever)

        results = strategy.retrieve_max_results('question', 2)

        # Assert that the results contain unique documents
        self.assertEqual(len(results.items), 2)
        self.assertEqual(set(item.document_id for item in results.items), {'doc1', 'doc2'})

    def test_extract_document_for_chunk_combines_all_chunks(self):
        # Mock the retriever's get_chunk method to return different chunks for a document
        self.retriever.find_relevant_chunks.return_value = [
            RelevantChunk('doc1', 0, 3, "This is the text for chunk 1 of 3", {'prop1': 'value1'}, 0.8),
        ]

        strategy = DocumentRetrievalStrategy(self.retriever, observe=False)

        result = strategy.retrieve_max_results(question='question', max_results=1)

        # Assert that the result's text is the combination of all chunks' text
        self.assertEqual(result.construct_context(), 'This is the text for chunk 1 of 3 This is the text for chunk 2 of 3 This is the text for chunk 3 of 3 \nprop1: value1 ')


if __name__ == '__main__':
    unittest.main()