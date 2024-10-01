import unittest
from unittest.mock import MagicMock

from rag4p.rag.model.chunk import Chunk
from rag4p.rag.retrieval.strategies.hierarchical_retrieval_strategy import HierarchicalRetrievalStrategy
from rag4p.rag.retrieval.retriever import Retriever
from rag4p.rag.model.relevant_chunk import RelevantChunk
from rag4p.rag.retrieval.retrieval_output import RetrievalOutput, RetrievalOutputItem

class TestHierarchicalRetrievalStrategy(unittest.TestCase):

    def setUp(self):
        self.retriever = MagicMock(spec=Retriever)
        self.retriever.find_relevant_chunks.return_value = [
            RelevantChunk(document_id="doc1", chunk_id="1_3_2", total_chunks=4, text="text1", properties={}, score=0.7)
        ]
        self.retriever.get_chunk.side_effect = self.mock_get_chunk
        self.strategy = HierarchicalRetrievalStrategy(self.retriever, max_levels=1)

    def mock_get_chunk(self, document_id, chunk_id):
        chunks = {
            "1_3_2": Chunk(document_id=document_id, chunk_id=chunk_id, chunk_text="hierarchical_text layer 3", total_chunks=4, properties={}),
            "1_3_3": Chunk(document_id=document_id, chunk_id=chunk_id, chunk_text="hierarchical_text layer 3 item 2", total_chunks=4, properties={}),
            "1_3": Chunk(document_id=document_id, chunk_id=chunk_id, chunk_text="hierarchical_text layer 2", total_chunks=2, properties={}),
            "1": Chunk(document_id=document_id, chunk_id=chunk_id, chunk_text="hierarchical_text layer 1", total_chunks=2, properties={})
        }
        return chunks.get(chunk_id)


    def test_retrieve_max_results_max_levels_1(self):

        strategy = HierarchicalRetrievalStrategy(self.retriever, max_levels=1)
        output = strategy.retrieve_max_results("question", 1)

        self.assertIsInstance(output, RetrievalOutput)
        self.assertEqual("doc1", output.items[0].document_id)
        self.assertEqual(1, len(output.items))
        self.assertEqual("1_3_2", output.items[0].chunk_id)
        self.assertEqual("hierarchical_text layer 2", output.items[0].text)

    def test_retrieve_max_results_max_levels_2(self):

        strategy = HierarchicalRetrievalStrategy(self.retriever, max_levels=2)
        output = strategy.retrieve_max_results("question", 1)

        self.assertIsInstance(output, RetrievalOutput)
        self.assertEqual(1, len(output.items))
        self.assertEqual("1_3_2", output.items[0].chunk_id)
        self.assertEqual("hierarchical_text layer 1", output.items[0].text)

    def test_retrieve_max_results_max_levels_to_high(self):

        strategy = HierarchicalRetrievalStrategy(self.retriever, max_levels=10)
        output = strategy.retrieve_max_results("question", 1)

        self.assertIsInstance(output, RetrievalOutput)
        self.assertEqual(1, len(output.items))
        self.assertEqual("1_3_2", output.items[0].chunk_id)
        self.assertEqual("hierarchical_text layer 1", output.items[0].text)

    def test_retrieve_max_results_max_levels_1_deduplicate(self):
        dedup_retriever = MagicMock(spec=Retriever)
        dedup_retriever.find_relevant_chunks.return_value = [
            RelevantChunk(document_id="doc1", chunk_id="1_3_2", total_chunks=4, text="text1", properties={}, score=0.7),
            RelevantChunk(document_id="doc1", chunk_id="1_3_3", total_chunks=4, text="text2", properties={}, score=0.6)
        ]
        dedup_retriever.get_chunk.side_effect = self.mock_get_chunk
        dedup_strategy = HierarchicalRetrievalStrategy(dedup_retriever, max_levels=1)

        output = dedup_strategy.retrieve_max_results("question", 1)

        self.assertIsInstance(output, RetrievalOutput)
        self.assertEqual(1, len(output.items))
        self.assertEqual("doc1", output.items[0].document_id)
        self.assertEqual("1_3_2", output.items[0].chunk_id)
        self.assertEqual("hierarchical_text layer 2", output.items[0].text)


if __name__ == '__main__':
    unittest.main()