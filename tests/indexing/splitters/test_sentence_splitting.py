import unittest

from rag4p.indexing.input_document import InputDocument
from rag4p.indexing.splitters.sentence_splitter import SentenceSplitter


class TestSentenceSplitter(unittest.TestCase):

    def setUp(self):
        self.splitter = SentenceSplitter()

    def test_splits_input_document_into_chunks(self):
        input_document = InputDocument(document_id='1', text='This is the first sentence. This is the second sentence.',
                                       properties={})
        chunks = self.splitter.split(input_document)
        self.assertEqual(len(chunks), 2)
        self.assertEqual(chunks[0].chunk_text, 'This is the first sentence.')
        self.assertEqual(chunks[1].chunk_text, 'This is the second sentence.')

    def test_splits_input_document_with_no_sentences(self):
        input_document = InputDocument(document_id='1', text='', properties={})
        chunks = self.splitter.split(input_document)
        self.assertEqual(len(chunks), 0)

    def test_splits_input_document_with_one_sentence(self):
        input_document = InputDocument(document_id='1', text='This is the only sentence.', properties={})
        chunks = self.splitter.split(input_document)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(chunks[0].chunk_text, 'This is the only sentence.')

    def test_splits_input_document_with_multiple_sentences(self):
        input_document = InputDocument(document_id='1', text='First sentence. Second sentence. Third sentence.',
                                       properties={})
        chunks = self.splitter.split(input_document)
        self.assertEqual(len(chunks), 3)
        self.assertEqual(chunks[0].chunk_text, 'First sentence.')
        self.assertEqual(chunks[1].chunk_text, 'Second sentence.')
        self.assertEqual(chunks[2].chunk_text, 'Third sentence.')


if __name__ == '__main__':
    unittest.main()
