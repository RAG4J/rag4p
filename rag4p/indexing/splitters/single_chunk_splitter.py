from rag4p.rag.model.chunk import Chunk
from rag4p.indexing.input_document import InputDocument
from rag4p.indexing.splitter import Splitter


class SingleChunkSplitter(Splitter):
    """
    Does not really split the document, it results in one chunk with the whole document.
    """

    def split(self, input_document: InputDocument) -> [str]:
        chunk = Chunk(input_document.document_id, 1, 1, input_document.text, input_document.properties)
        return [chunk]

    @staticmethod
    def name() -> str:
        return "SingleChunkSplitter"
