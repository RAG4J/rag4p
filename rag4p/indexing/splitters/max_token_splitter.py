from typing import List

import tiktoken

from rag4p.rag.model.chunk import Chunk
from rag4p.indexing.input_document import InputDocument
from rag4p.indexing.splitter import Splitter
from rag4p.integrations.openai import DEFAULT_EMBEDDING_MODEL


class MaxTokenSplitter(Splitter):
    """
    Splits an InputDocument into Chunks of a maximum number of tokens. The tokens are obtained by
    encoding the text of the document using the default model from openai encoding. The chunks of tokens are
    decoded back into text.
    """
    def __init__(self, max_tokens: int = 200, model: str = DEFAULT_EMBEDDING_MODEL):
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
