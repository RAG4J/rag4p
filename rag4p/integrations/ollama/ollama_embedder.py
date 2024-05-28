from rag4p.integrations.ollama import DEFAULT_EMBEDDING_MODEL
from rag4p.integrations.ollama.access_ollama import AccessOllama
from rag4p.rag.embedding.embedder import Embedder


class OllamaEmbedder(Embedder):
    """
    Embedder that uses Ollama's API to embed text.
    """

    def __init__(self, access_ollama: AccessOllama, model: str = DEFAULT_EMBEDDING_MODEL):
        self.ollama = access_ollama
        self.embedding_model = model

    def model(self) -> str:
        return self.embedding_model

    def identifier(self) -> str:
        return f"{self.supplier().lower()}-embedder-{self.embedding_model.lower()}"

    def embed(self, text: str) -> [float]:
        return self.ollama.generate_embedding(text, model=self.embedding_model)

    @staticmethod
    def supplier() -> str:
        return "Ollama"
