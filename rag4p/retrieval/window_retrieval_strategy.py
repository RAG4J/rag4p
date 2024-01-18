from rag4p.domain.relevant_chunk import RelevantChunk
from rag4p.domain.retrieval_output import RetrievalOutput, RetrievalOutputItem
from rag4p.retrieval.retrieval_strategy import RetrievalStrategy
from rag4p.retrieval.retriever import Retriever
from rag4p.tracker.rag_tracker import global_data


class WindowRetrievalStrategy(RetrievalStrategy):

    def __init__(self, retriever: Retriever, window_size: int = 1):
        self.retriever = retriever
        self.window_size = window_size

    def retrieve_max_results(self, question, max_results) -> RetrievalOutput:
        relevant_chunks = self.retriever.find_relevant_chunks(question, max_results)

        return self.__extract_window_for_relevant_chunks(relevant_chunks)

    def retrieve_max_results_observed(self, question, max_results) -> RetrievalOutput:
        relevant_chunks = self.retriever.find_relevant_chunks(question, max_results)

        return self.__extract_window_for_relevant_chunks(relevant_chunks, observe=True)

    def __extract_window_for_relevant_chunks(self, relevant_chunks: [RelevantChunk], observe: bool = False) \
            -> RetrievalOutput:
        retrieval_output_items = []
        for relevant_chunk in relevant_chunks:
            chunk_ids = self.__chunk_ids_for_window(relevant_chunk.chunk_id,
                                                    self.window_size,
                                                    relevant_chunk.total_chunks)
            overall_text = ""
            for chunk_id in chunk_ids:
                chunk = self.retriever.get_chunk(relevant_chunk.document_id, chunk_id)
                overall_text += chunk.chunk_text + " "

            if observe:
                global_data["observer"].add_relevant_chunk(relevant_chunk.get_id(), overall_text)

            relevant_item = RetrievalOutputItem(document_id=relevant_chunk.document_id,
                                                chunk_id=relevant_chunk.chunk_id,
                                                text=overall_text)
            retrieval_output_items.append(relevant_item)
        return RetrievalOutput(retrieval_output_items)

    @staticmethod
    def __chunk_ids_for_window(chunk_id: int, window_size: int, number_of_chunks: int) -> [int]:
        # Calculate the start and end of the window
        start = max(0, chunk_id - window_size)
        end = min(number_of_chunks - 1, chunk_id + window_size)

        # Generate the list of chunk_ids
        chunk_ids = [start + i for i in range(end - start + 1)]

        return chunk_ids
