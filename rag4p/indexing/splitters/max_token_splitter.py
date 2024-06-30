from typing import List

import tiktoken
from tokenizers import Tokenizer

from rag4p.integrations.ollama.ollama_tokenizer import tokenizer_for_model
from rag4p.rag.model.chunk import Chunk
from rag4p.indexing.input_document import InputDocument
from rag4p.indexing.splitter import Splitter
from rag4p.integrations.openai import DEFAULT_EMBEDDING_MODEL, PROVIDER as OPENAI_PROVIDER
from rag4p.integrations.ollama import PROVIDER as OLLAMA_PROVIDER
from rag4p.integrations.bedrock import PROVIDER as BEDROCK_PROVIDER


class MaxTokenSplitter(Splitter):
    """
    Splits an InputDocument into Chunks of a maximum number of tokens. The tokens are obtained by
    encoding the text of the document using the default model from openai encoding. The chunks of tokens are
    decoded back into text.
    """

    def __init__(self, max_tokens: int = 200, provider: str = OPENAI_PROVIDER, model: str = DEFAULT_EMBEDDING_MODEL):
        self.max_tokens = max_tokens
        self.provider = provider
        if provider == OPENAI_PROVIDER:
            self.encoding = tiktoken.encoding_for_model(model)
        elif provider == OLLAMA_PROVIDER:
            tokenize_model = tokenizer_for_model(model)
            self.encoding = Tokenizer.from_pretrained(tokenize_model)
        elif provider == BEDROCK_PROVIDER:
            # TODO: Implement Bedrock tokenizer, but no information found for the moment, so abuse OpenAI tokenizer
            self.encoding = tiktoken.encoding_for_model(DEFAULT_EMBEDDING_MODEL)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def split(self, input_document: InputDocument) -> List[Chunk]:
        if self.provider == OPENAI_PROVIDER:
            tokens = self.encoding.encode(input_document.text)
        elif self.provider == OLLAMA_PROVIDER:
            tokens = self.encoding.encode(input_document.text).ids
        elif self.provider == BEDROCK_PROVIDER:
            tokens = self.encoding.encode(input_document.text)
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")

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

    @staticmethod
    def name() -> str:
        return "MaxTokenSplitter"
