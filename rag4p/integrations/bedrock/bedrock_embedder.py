from rag4p.integrations.bedrock import DEFAULT_EMBEDDING_MODEL, PROVIDER
from rag4p.integrations.bedrock.access_bedrock import AccessBedrock
from rag4p.rag.embedding.embedder import Embedder


class BedrockEmbedder(Embedder):
    """
    Embedder that uses Bedrock to embed text.
    """

    def __init__(self, access_bedrock: AccessBedrock, model: str = DEFAULT_EMBEDDING_MODEL):
        self.bedrock = access_bedrock
        self.embedding_model = model

    def model(self) -> str:
        return self.embedding_model

    def identifier(self) -> str:
        return f"{self.supplier().lower()}-embedder-{self.embedding_model.lower()}"

    def embed(self, text: str) -> [float]:
        return self.bedrock.generate_embedding(text, model=self.embedding_model)

    @staticmethod
    def supplier() -> str:
        return PROVIDER
