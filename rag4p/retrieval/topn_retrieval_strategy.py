from rag4p.domain.retrieval_output import RetrievalOutput, RetrievalOutputItem
from rag4p.retrieval.retrieval_strategy import RetrievalStrategy
from rag4p.retrieval.retriever import Retriever


class TopNRetrievalStrategy(RetrievalStrategy):

    def __init__(self, retriever: Retriever):
        self.retriever = retriever

    def retrieve_max_results(self, question: str, max_results: int) -> RetrievalOutput:
        return self.__find_relevant_chunks(question, max_results)

    def retrieve_max_results_observed(self, question: str, max_results: int) -> RetrievalOutput:
        # All observation is done in the ObservedRetriever, use that to observe.
        return self.__find_relevant_chunks(question, max_results)

    def __find_relevant_chunks(self, question: str, max_results: int) -> RetrievalOutput:
        relevant_chunks = self.retriever.find_relevant_chunks(question, max_results)
        retrieval_output_items = []
        for relevant_chunk in relevant_chunks:
            retrieval_output_items.append(RetrievalOutputItem(document_id=relevant_chunk.document_id,
                                                              chunk_id=relevant_chunk.chunk_id,
                                                              text=relevant_chunk.chunk_text))
        return RetrievalOutput(retrieval_output_items)
