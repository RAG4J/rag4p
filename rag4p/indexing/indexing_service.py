from rag4p.indexing.content_reader import ContentReader
from rag4p.indexing.content_store import ContentStore
from rag4p.indexing.splitter import Splitter


class IndexingService():
    def __init__(self, content_store: ContentStore):
        self.content_store = content_store

    def index_documents(self, content_reader: ContentReader, splitter: Splitter):
        for document in content_reader.read():
            self.content_store.store_document(document, splitter)