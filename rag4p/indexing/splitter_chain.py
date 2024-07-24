from abc import ABC
from typing import List

from rag4p.indexing.input_document import InputDocument
from rag4p.indexing.splitter import Splitter
from rag4p.rag.model.chunk import Chunk


class SplitterChain(Splitter):

    def __init__(self, splitters: List[Splitter], include_all_chunks: bool = False):
        self.splitters = splitters
        if not self.splitters or len(self.splitters) == 0:
            raise ValueError("At least one splitter must be provided.")
        self.include_all_chunks = include_all_chunks

    def split(self, input_document: InputDocument, parent_chunk: Chunk = None) -> List[Chunk]:
        return self._split_current_splitter(input_document=input_document)

    def _split_current_splitter(self,
                                input_document: InputDocument,
                                splitter_nr: int = 0,
                                parent_chunk: Chunk = None) -> List[Chunk]:
        chunks = []
        if parent_chunk is None:
            print("No parent chunk")
        else:
            print(f"Parent chunk text: {parent_chunk.chunk_text}")

        current_chunks = self.splitters[splitter_nr].split(input_document=input_document, parent_chunk=parent_chunk)
        for current_chunk in current_chunks:

            # We always add the first and last chunk, or all chunks if include_all_chunks is True
            if splitter_nr + 1 >= len(self.splitters) or self.include_all_chunks or splitter_nr == 0:
                chunks.append(current_chunk)

            # If we are not at the last splitter, we recursively call the next splitter
            if splitter_nr + 1 < len(self.splitters):
                child_chunks = self._split_current_splitter(input_document=input_document,
                                                            splitter_nr=splitter_nr + 1,
                                                            parent_chunk=current_chunk)
                chunks.extend(child_chunks)

        return chunks

    @staticmethod
    def name() -> str:
        return "SplitterChain"
