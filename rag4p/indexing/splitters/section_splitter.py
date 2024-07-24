import re
from typing import List

from rag4p.indexing.input_document import InputDocument
from rag4p.indexing.splitter import Splitter
from rag4p.rag.model.chunk import Chunk


class SectionSplitter(Splitter):
    def split(self, input_document: InputDocument, parent_chunk: Chunk = None) -> List[Chunk]:
        input_text = input_document.text if parent_chunk is None else parent_chunk.chunk_text
        sections = re.split(r"\n\s*\n", input_text)

        chunks_ = []
        for i, section in enumerate(sections):
            chunk_id = str(i) if parent_chunk is None else f"{parent_chunk.chunk_id}_{i}"
            chunk_ = Chunk(input_document.document_id, chunk_id, len(sections), section, input_document.properties)
            chunks_.append(chunk_)

        return chunks_

    @staticmethod
    def name() -> str:
        return SectionSplitter.__name__
