from abc import ABC, abstractmethod
from typing import List

from rag4p.domain.chunk import Chunk
from rag4p.domain.input_document import InputDocument
from rag4p.indexing.embedder import Embedder
from rag4p.indexing.splitter import Splitter


class ContentStore(ABC):
    embedder: Embedder = None

    def __init__(self, embedder: Embedder):
        self.embedder = embedder

    @abstractmethod
    def store(self, chunks: List[Chunk]):
        pass

    def store_document(self, document: InputDocument, splitter: Splitter):
        chunks = splitter.split(document)
        self.store(chunks)
