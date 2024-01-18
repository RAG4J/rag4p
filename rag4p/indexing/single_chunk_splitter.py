from rag4p.domain.chunk import Chunk
from rag4p.domain.input_document import InputDocument
from rag4p.indexing.splitter import Splitter


class SingleChunkSplitter(Splitter):

    def split(self, input_document: InputDocument) -> [str]:
        chunk = Chunk(input_document.document_id, 1, 1, input_document.text, input_document.properties)
        return [chunk]
