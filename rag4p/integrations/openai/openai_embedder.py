from openai import OpenAI

from rag4p.integrations.openai import EMBEDDING_SMALL
from rag4p.rag.embedding.embedder import Embedder


class OpenAIEmbedder(Embedder):
    """
    Embedder that uses OpenAI's API to embed text. Look at the documentation for more information about the models,
    https://platform.openai.com/docs/models/embeddings. At the moment the default model is text-embedding-3-small.
    Before that text-embedding-ada-002 was used.
    """

    def __init__(self, api_key: str, embedding_model: str = EMBEDDING_SMALL):
        self.client = OpenAI(
            # This is the default and can be omitted
            api_key=api_key,
        )

        self.embedding_model = embedding_model

    def identifier(self) -> str:
        return f"openai-embedder-{self.embedding_model}"

    def embed(self, text: str) -> [float]:
        response = self.client.embeddings.create(input=text, model=self.embedding_model, encoding_format="float")
        embeddings = response.data

        return embeddings[0].embedding
