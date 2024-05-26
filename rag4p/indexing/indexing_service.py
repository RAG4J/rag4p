from abc import ABC

from rag4p.indexing.content_reader import ContentReader
from rag4p.indexing.input_document import InputDocument
from rag4p.rag.store.content_store import ContentStore
from rag4p.indexing.splitter import Splitter


class IndexingService(ABC):
    def __init__(self, content_store: ContentStore):
        self.content_store = content_store

    def index_documents(self, content_reader: ContentReader, splitter: Splitter):
        for batch in content_reader.read():
            print(f"Indexing batch of size {len(batch)}")
            for document in batch:
                self.index_document(document, splitter)

    def index_document(self, document: InputDocument, splitter: Splitter):
        chunks = splitter.split(document)
        self.content_store.store(chunks)