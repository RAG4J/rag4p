from abc import ABC, abstractmethod

from rag4p.rag.retrieval.retrieval_output import RetrievalOutput


class RetrievalStrategy(ABC):

    def retrieve(self, question: str) -> RetrievalOutput:
        return self.retrieve_max_results(question, 4)

    @abstractmethod
    def retrieve_max_results(self, question: str, max_results: int) -> RetrievalOutput:
        pass

    @abstractmethod
    def retrieve_max_results_observed(self, question: str, max_results: int) -> RetrievalOutput:
        pass
