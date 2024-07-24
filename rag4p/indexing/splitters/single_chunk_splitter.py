from rag4p.indexing.input_document import InputDocument
from rag4p.indexing.splitter import Splitter
from rag4p.rag.model.chunk import Chunk


class SingleChunkSplitter(Splitter):
    """
    Does not really split the document, it results in one chunk with the whole document.
    """

    def split(self, input_document: InputDocument, parent_chunk: Chunk = None) -> [Chunk]:
        chunk_id = "0" if parent_chunk is None else f"{parent_chunk.chunk_id}_0"
        chunk_text = input_document.text if parent_chunk is None else parent_chunk.chunk_text
        chunk = Chunk(input_document.document_id, chunk_id, 1, chunk_text, input_document.properties)
        return [chunk]

    @staticmethod
    def name() -> str:
        return "SingleChunkSplitter"
