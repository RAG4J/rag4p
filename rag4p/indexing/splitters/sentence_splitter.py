from rag4p.indexing.input_document import InputDocument
from rag4p.indexing.splitter import Splitter
from rag4p.rag.model.chunk import Chunk
from nltk.tokenize import sent_tokenize
from typing import List


class SentenceSplitter(Splitter):
    """
    Splits an InputDocument into Chunks of a single sentence.
    """

    def split(self, input_document: InputDocument) -> List[Chunk]:
        sentences = sent_tokenize(input_document.text)
        chunks = []
        for i in range(len(sentences)):
            chunk = Chunk(input_document.document_id, i, len(sentences), sentences[i], input_document.properties)
            chunks.append(chunk)
        return chunks
