import unittest

from rag4p.connectopenai import DEFAULT_MODEL
from rag4p.connectopenai.max_token_splitter import MaxTokenSplitter
from rag4p.domain.input_document import InputDocument


class TestMaxTokenSplitter(unittest.TestCase):

    def setUp(self):
        self.splitter = MaxTokenSplitter(max_tokens=5, model=DEFAULT_MODEL)

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


if __name__ == '__main__':
    unittest.main()