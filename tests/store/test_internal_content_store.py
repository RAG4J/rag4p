import unittest
from unittest.mock import patch, MagicMock
from rag4p.domain.chunk import Chunk
from rag4p.store.internal_content_store import InternalContentStore
from rag4p.indexing.embedder import Embedder


class TestInternalContentStore(unittest.TestCase):

    @patch.object(Embedder, 'embed')
    def test_stores_chunk_correctly(self, mock_embed):
        mock_embed.embed.return_value = [0.1, 0.2, 0.3]
        store = InternalContentStore(mock_embed)

        chunk = Chunk(document_id='1', chunk_id=1, chunk_text='This is a chunk.', total_chunks=1, properties={})
        store.store([chunk])
        self.assertEqual(1, len(store.df))
        self.assertEqual('1_1', store.df.iloc[0]['chunk_id'])
        self.assertEqual('This is a chunk.', store.df.iloc[0]['chunk'].chunk_text)
        self.assertEqual([0.1, 0.2, 0.3], store.df.iloc[0]['embedding'])

    @patch.object(Embedder, 'embed')
    def test_finds_relevant_chunks(self, mock_embed):
        mock_embed.embed.return_value = [0.1, 0.2, 0.3]
        store = InternalContentStore(mock_embed)
        chunk1 = Chunk(document_id='1', chunk_id=1, chunk_text='This is the first chunk.', total_chunks=1, properties={})
        chunk2 = Chunk(document_id='2', chunk_id=2, chunk_text='This is the second chunk.', total_chunks=1, properties={})
        store.store([chunk1, chunk2])
        relevant_chunks = store.find_relevant_chunks('This is a query.')
        self.assertEqual(2, len(relevant_chunks))

    @patch.object(Embedder, 'embed')
    def test_gets_chunk_by_id(self, mock_embed):
        mock_embed.embed.return_value = [0.1, 0.2, 0.3]
        store = InternalContentStore(mock_embed)
        chunk = Chunk(document_id='1', chunk_id=1, chunk_text='This is a chunk.', total_chunks=1, properties={})
        store.store([chunk])
        retrieved_chunk = store.get_chunk('1_1')
        self.assertEqual('1_1', retrieved_chunk.get_id())
        self.assertEqual('This is a chunk.', retrieved_chunk.chunk_text)

    @patch.object(Embedder, 'embed')
    def test_gets_chunk_by_non_existing_id(self, mock_embed):
        mock_embed.embed.return_value = [0.1, 0.2, 0.3]
        store = InternalContentStore(mock_embed)
        chunk = Chunk(document_id='1', chunk_id=1, chunk_text='This is a chunk.', total_chunks=1, properties={})
        store.store([chunk])
        with self.assertRaises(Exception):
            store.get_chunk('1_2')


if __name__ == '__main__':
    unittest.main()
