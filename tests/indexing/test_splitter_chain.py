import unittest
from unittest.mock import MagicMock
from rag4p.indexing.splitter_chain import SplitterChain
from rag4p.indexing.input_document import InputDocument
from rag4p.indexing.splitter import Splitter
from rag4p.indexing.splitters.max_token_splitter import MaxTokenSplitter
from rag4p.indexing.splitters.sentence_splitter import SentenceSplitter
from rag4p.indexing.splitters.single_chunk_splitter import SingleChunkSplitter
from rag4p.rag.model.chunk import Chunk


class TestSplitterChain(unittest.TestCase):

    def test_splitter_chain_initializes_with_valid_splitters(self):
        splitter = SingleChunkSplitter()
        splitter_chain = SplitterChain([splitter])
        self.assertEqual(len(splitter_chain.splitters), 1)

    def test_splitter_chain_raises_error_with_no_splitters(self):
        with self.assertRaises(ValueError):
            SplitterChain([])

    def test_splitter_chain_splits_document_correctly(self):
        splitter = SentenceSplitter()
        input_document = InputDocument(text="Dit is zin 1. En zin 2.", properties={}, document_id="doc1")
        splitter_chain = SplitterChain([splitter])
        chunks = splitter_chain.split(input_document)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].chunk_text, "Dit is zin 1.")
        self.assertEqual(chunks[1].chunk_text, "En zin 2.")

    def test_splitter_chain_handles_multiple_splitters(self):
        input_document = InputDocument(text="Dit is zin 1. En zin 2. Laatste is 3.", properties={}, document_id="doc1")
        splitter_chain = SplitterChain(
            [SingleChunkSplitter(), MaxTokenSplitter(max_tokens=14), SentenceSplitter()],
            include_all_chunks=True)
        chunks = splitter_chain.split(input_document)
        self.assertEqual(6, len(chunks))
        self.assertEqual(chunks[0].chunk_text, "Dit is zin 1. En zin 2. Laatste is 3.")
        self.assertEqual(chunks[0].chunk_id, "0")
        self.assertEqual(chunks[0].get_id(), "doc1_0")
        self.assertEqual(chunks[1].chunk_text, "Dit is zin 1. En zin 2.")
        self.assertEqual(chunks[1].chunk_id, "0_0")
        self.assertEqual(chunks[1].get_id(), "doc1_0_0")
        self.assertEqual(chunks[2].chunk_text, "Dit is zin 1.")
        self.assertEqual(chunks[2].chunk_id, "0_0_0")
        self.assertEqual(chunks[2].get_id(), "doc1_0_0_0")
        self.assertEqual(chunks[3].chunk_text, "En zin 2.")
        self.assertEqual(chunks[3].chunk_id, "0_0_1")
        self.assertEqual(chunks[3].get_id(), "doc1_0_0_1")
        self.assertEqual(chunks[4].chunk_text, " Laatste is 3.")
        self.assertEqual(chunks[4].chunk_id, "0_1")
        self.assertEqual(chunks[4].get_id(), "doc1_0_1")
        self.assertEqual(chunks[5].chunk_text, " Laatste is 3.")
        self.assertEqual(chunks[5].chunk_id, "0_1_0")
        self.assertEqual(chunks[5].get_id(), "doc1_0_1_0")

    def test_splitter_chain_handles_multiple_splitters_include_all_false(self):
        input_document = InputDocument(text="Dit is zin 1. En zin 2. Laatste is 3.", properties={}, document_id="doc1")
        splitter_chain = SplitterChain(
            [SingleChunkSplitter(), MaxTokenSplitter(max_tokens=14), SentenceSplitter()],
            include_all_chunks=False)
        chunks = splitter_chain.split(input_document)
        self.assertEqual(4, len(chunks))
        self.assertEqual(chunks[0].chunk_text, "Dit is zin 1. En zin 2. Laatste is 3.")
        self.assertEqual(chunks[0].chunk_id, "0")
        self.assertEqual(chunks[0].get_id(), "doc1_0")
        self.assertEqual(chunks[1].chunk_text, "Dit is zin 1.")
        self.assertEqual(chunks[1].chunk_id, "0_0_0")
        self.assertEqual(chunks[1].get_id(), "doc1_0_0_0")
        self.assertEqual(chunks[2].chunk_text, "En zin 2.")
        self.assertEqual(chunks[2].chunk_id, "0_0_1")
        self.assertEqual(chunks[2].get_id(), "doc1_0_0_1")
        self.assertEqual(chunks[3].chunk_text, " Laatste is 3.")
        self.assertEqual(chunks[3].chunk_id, "0_1_0")
        self.assertEqual(chunks[3].get_id(), "doc1_0_1_0")

    def test_splitter_chain_handles_empty_text(self):
        splitter = MagicMock(spec=Splitter)
        splitter.split.return_value = []
        input_document = InputDocument(text="", properties={}, document_id="doc1")
        splitter_chain = SplitterChain([splitter])
        chunks = splitter_chain.split(input_document)
        self.assertEqual(len(chunks), 0)


if __name__ == '__main__':
    unittest.main()
