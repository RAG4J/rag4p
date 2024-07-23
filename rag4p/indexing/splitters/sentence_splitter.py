from typing import List

from nltk.tokenize import sent_tokenize

from rag4p.indexing.input_document import InputDocument
from rag4p.indexing.splitter import Splitter
from rag4p.rag.model.chunk import Chunk


class SentenceSplitter(Splitter):
    """
    Splits an InputDocument into Chunks of a single sentence.
    """

    def split(self, input_document: InputDocument, parent_chunk: Chunk = None) -> List[Chunk]:
        input_text = input_document.text if parent_chunk is None else parent_chunk.chunk_text
        sentences = sent_tokenize(input_text)
        chunks = []
        for i in range(len(sentences)):
            chunk_id = str(i) if parent_chunk is None else f"{parent_chunk.chunk_id}_{i}"
            chunk = Chunk(input_document.document_id, chunk_id, len(sentences), sentences[i], input_document.properties)
            chunks.append(chunk)
        return chunks

    @staticmethod
    def name() -> str:
        return "SentenceSplitter"
