from openai import OpenAI

from rag4p.indexing.embedder import Embedder


class OpenAIEmbedder(Embedder):
    def __init__(self, api_key: str):
        self.client = OpenAI(
            # This is the default and can be omitted
            api_key=api_key,
        )

    def embed(self, text: str) -> [float]:
        response = self.client.embeddings.create(input=text,model="text-embedding-ada-002", encoding_format="float")
        embeddings = response.data

        return embeddings[0].embedding