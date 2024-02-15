from rag4p.indexing.content_reader import ContentReader
from rag4p.indexing.input_document import InputDocument
from rag4p.rag.store.content_store import ContentStore
from rag4p.indexing.splitter import Splitter


class IndexingService():
    def __init__(self, content_store: ContentStore):
        self.content_store = content_store

    def index_documents(self, content_reader: ContentReader, splitter: Splitter):
        for document in content_reader.read():
            self.index_document(document, splitter)

    def index_document(self, document: InputDocument, splitter: Splitter):
        chunks = splitter.split(document)
        self.content_store.store(chunks)