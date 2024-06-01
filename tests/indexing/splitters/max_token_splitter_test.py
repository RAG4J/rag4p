import unittest

from rag4p.indexing.input_document import InputDocument
from rag4p.indexing.splitters.max_token_splitter import MaxTokenSplitter
from rag4p.integrations.ollama import EMBEDDING_MODEL_NOMIC
from rag4p.integrations.ollama import PROVIDER as OLLAMA_PROVIDER
from rag4p.integrations.openai import DEFAULT_EMBEDDING_MODEL
from rag4p.integrations.openai import PROVIDER as OPENAI_PROVIDER


class TestMaxTokenSplitter(unittest.TestCase):

    def setUp(self):
        self.splitter = MaxTokenSplitter(max_tokens=5, provider=OPENAI_PROVIDER, model=DEFAULT_EMBEDDING_MODEL)

    def test_split_into_chunks_with_exact_multiple_of_max_tokens(self):
        input_document = InputDocument(document_id="1", text="This is a test document", properties={})
        chunks = self.splitter.split(input_document)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].total_chunks, 1)

    def test_split_into_chunks_with_less_than_max_tokens(self):
        self.splitter.max_tokens = 10
        input_document = InputDocument(document_id="1", text="Short text.", properties={})
        chunks = self.splitter.split(input_document)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].total_chunks, 1)

    def test_split_into_chunks_with_more_than_max_tokens(self):
        input_document = InputDocument(document_id="1", text="This is a longer test document with more than max tokens.", properties={})
        chunks = self.splitter.split(input_document)
        self.assertGreater(len(chunks), 1)
        self.assertTrue(all(chunk.total_chunks == len(chunks) for chunk in chunks))

    def test_split_into_chunks_with_no_tokens(self):
        input_document = InputDocument(document_id="1", text="", properties={})
        chunks = self.splitter.split(input_document)
        self.assertEqual(len(chunks), 0)

    def test_split_into_chunks_with_ollama_provider(self):
        """Note: The nomic tokenizer uses special tokens that are also counted in the token count. Therefore, the
        total amount of chunks is 2 instead of 1."""
        splitter = MaxTokenSplitter(max_tokens=5, provider=OLLAMA_PROVIDER, model=EMBEDDING_MODEL_NOMIC)
        input_document = InputDocument(document_id="1", text="This is a test document", properties={})
        chunks = splitter.split(input_document)
        self.assertEqual(2, len(chunks))
        self.assertEqual(2, chunks[0].total_chunks)


if __name__ == '__main__':
    unittest.main()
