import unittest
from unittest.mock import MagicMock

from rag4p.rag.model.chunk import Chunk
from rag4p.rag.retrieval.strategies.window_retrieval_strategy import WindowRetrievalStrategy
from rag4p.rag.retrieval.retriever import Retriever
from rag4p.rag.model.relevant_chunk import RelevantChunk
from rag4p.rag.retrieval.retrieval_output import RetrievalOutput, RetrievalOutputItem
from rag4p.rag.tracker.rag_tracker import global_data


class TestWindowRetrievalStrategy(unittest.TestCase):

    def test_windowRetrievalStrategy_initializes_with_valid_retriever_and_window_size(self):
        retriever = MagicMock(spec=Retriever)
        strategy = WindowRetrievalStrategy(retriever, window_size=2)
        self.assertEqual(strategy.window_size, 2)
        self.assertEqual(strategy.retriever, retriever)

    def test_windowRetrievalStrategy_retrieve_max_results_returns_correct_output(self):
        retriever = MagicMock(spec=Retriever)
        retriever.find_relevant_chunks.return_value = [
            RelevantChunk(document_id="doc1", chunk_id="0", text="text1", total_chunks=3, properties={}, score=0.8),
        ]
        retriever.get_chunk.side_effect = [
            Chunk(document_id="doc1", chunk_id="0", chunk_text="text1", total_chunks=3, properties={}),
            Chunk(document_id="doc1", chunk_id="1", chunk_text="text2", total_chunks=3, properties={}),
        ]
        strategy = WindowRetrievalStrategy(retriever, window_size=1)
        output = strategy.retrieve_max_results("question", 1)
        self.assertIsInstance(output, RetrievalOutput)
        self.assertEqual(len(output.items), 1)
        self.assertEqual(output.items[0].text, "text1 text2 ")

    def test_windowRetrievalStrategy_retrieve_max_results_returns_correct_output_not_enough_chunks(self):
        retriever = MagicMock(spec=Retriever)
        retriever.find_relevant_chunks.return_value = [
            RelevantChunk(document_id="doc1", chunk_id="1", text="text1", total_chunks=2, properties={}, score=0.8),
        ]
        retriever.get_chunk.side_effect = [
            Chunk(document_id="doc1", chunk_id="0", chunk_text="text1", total_chunks=2, properties={}),
            Chunk(document_id="doc1", chunk_id="1", chunk_text="text2", total_chunks=2, properties={}),
        ]
        strategy = WindowRetrievalStrategy(retriever, window_size=1)
        output = strategy.retrieve_max_results("question", 1)
        self.assertIsInstance(output, RetrievalOutput)
        self.assertEqual(len(output.items), 1)
        self.assertEqual(output.items[0].text, "text1 text2 ")

    def test_windowRetrievalStrategy_retrieve_max_results_returns_correct_output_window_both_sides(self):
        retriever = MagicMock(spec=Retriever)
        retriever.find_relevant_chunks.return_value = [
            RelevantChunk(document_id="doc1", chunk_id="1", text="text1", total_chunks=3, properties={}, score=0.8),
        ]
        retriever.get_chunk.side_effect = [
            Chunk(document_id="doc1", chunk_id="0", chunk_text="text1", total_chunks=3, properties={}),
            Chunk(document_id="doc1", chunk_id="1", chunk_text="text2", total_chunks=3, properties={}),
            Chunk(document_id="doc1", chunk_id="2", chunk_text="text3", total_chunks=3, properties={}),
        ]
        strategy = WindowRetrievalStrategy(retriever, window_size=1)
        output = strategy.retrieve_max_results("question", 1)
        self.assertIsInstance(output, RetrievalOutput)
        self.assertEqual(len(output.items), 1)
        self.assertEqual(output.items[0].text, "text1 text2 text3 ")

    def test_windowRetrievalStrategy_retrieve_max_results_observed_adds_to_observer(self):
        retriever = MagicMock(spec=Retriever)
        retriever.find_relevant_chunks.return_value = [
            RelevantChunk(document_id="doc1", chunk_id="0", text="text1", total_chunks=3, properties={}, score=0.8),
        ]
        retriever.get_chunk.side_effect = [
            Chunk(document_id="doc1", chunk_id="0", chunk_text="text1", total_chunks=3, properties={}),
            Chunk(document_id="doc1", chunk_id="1", chunk_text="text2", total_chunks=3, properties={}),
        ]
        global_data["observer"] = MagicMock()
        strategy = WindowRetrievalStrategy(retriever, window_size=1)
        strategy.retrieve_max_results_observed("question", 1)
        global_data["observer"].add_relevant_chunk.assert_called_once_with("doc1_0", "text1 text2 ")


if __name__ == '__main__':
    unittest.main()
