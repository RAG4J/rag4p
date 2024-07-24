import time
from abc import ABC

from rag4p.indexing.content_reader import ContentReader
from rag4p.indexing.indexing_response import IndexingResponse
from rag4p.indexing.input_document import InputDocument
from rag4p.indexing.splitter import Splitter
from rag4p.rag.store.content_store import ContentStore


class IndexingService(ABC):
    def __init__(self, content_store: ContentStore):
        self.content_store = content_store

    def index_documents(self, content_reader: ContentReader, splitter: Splitter) -> IndexingResponse:
        start_time = time.time()
        num_documents = 0
        num_chunks = 0
        for batch in content_reader.read():
            num_documents += len(batch)
            print(f"Indexing batch of size {len(batch)}")
            for document in batch:
                doc_chunks = self.index_document(document, splitter)
                num_chunks += doc_chunks

        end_time = time.time()
        response_time = end_time - start_time

        return IndexingResponse(
            num_documents=num_documents,
            num_chunks=num_chunks,
            content_reader=content_reader.name(),
            splitter=splitter.name(),
            running_time=response_time
        )

    def index_document(self, document: InputDocument, splitter: Splitter) -> int:
        chunks = splitter.split(document)
        self.content_store.store(chunks)
        return len(chunks)