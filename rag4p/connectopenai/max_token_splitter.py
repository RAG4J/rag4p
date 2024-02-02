from typing import List

import tiktoken

from rag4p.connectopenai import DEFAULT_MODEL
from rag4p.domain.chunk import Chunk
from rag4p.domain.input_document import InputDocument
from rag4p.indexing.splitter import Splitter


class MaxTokenSplitter(Splitter):
    def __init__(self, max_tokens: int = 200, model: str = DEFAULT_MODEL):
        self.max_tokens = max_tokens
        self.encoding = tiktoken.encoding_for_model(model)

    def split(self, input_document: InputDocument) -> List[Chunk]:
        tokens = self.encoding.encode(input_document.text)

        chunks = []
        num_chunks = len(tokens) // self.max_tokens + (len(tokens) % self.max_tokens != 0)

        while tokens:
            chunk_size = min(len(tokens), self.max_tokens)
            chunk_tokens = tokens[:chunk_size]
            tokens = tokens[chunk_size:]
            chunk_text = self.encoding.decode(chunk_tokens)
            chunk = Chunk(input_document.document_id, len(chunks), num_chunks, chunk_text, input_document.properties)
            chunks.append(chunk)

        return chunks
