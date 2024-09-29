from rag4p.rag.model.relevant_chunk import RelevantChunk
from rag4p.rag.retrieval.retrieval_output import RetrievalOutput, RetrievalOutputItem
from rag4p.rag.retrieval.retrieval_strategy import RetrievalStrategy
from rag4p.rag.retrieval.retriever import Retriever
from rag4p.rag.tracker.rag_tracker import global_data


class HierarchicalRetrievalStrategy(RetrievalStrategy):
    """
    This retrieval strategy works with hierarchical chunks. It retrieves the relevant chunks and then retrieves the
    chunks higher in the hierarchy of the relevant chunks. For example, if chunks are created using a SplitterChain.
    Each chunk is related to the document, or the chunk created from the filter higher up in the chain.

    The Chain: SectionSplitter -> SentenceSplitter
    The ids for the chunks are: docId_chunkIdSplitter1_chunkIdSplitter2

    Now a retriever finds the relevant chunk doc1_1_3. This means that the chunk with id 3 from the SentenceSplitter
    is relevant. Counting starts from 0. This strategy will retrieve the chunk with id 1 from the SectionSplitter.

    When the SplitterChain creates more levels, you can configure the maximum amount of levels to go up in the
    hierarchy. If we have 5 levels in the chain, and the max_levels is set to 2, we find a relevant chunk with id
    doc1_1_3_2_1. This means that the chunk with id doc1_1_3 is passed for the context.
    """

    def __init__(self, retriever: Retriever, max_levels: int = 1):
        self.retriever = retriever
        self.max_levels = max_levels

    def retrieve_max_results(self, question: str, max_results: int) -> RetrievalOutput:
        relevant_chunks = self.retriever.find_relevant_chunks(question, max_results)

        return self.__extract_hierarchy_for_chunks(relevant_chunks)

    def retrieve_max_results_observed(self, question: str, max_results: int) -> RetrievalOutput:
        relevant_chunks = self.retriever.find_relevant_chunks(question, max_results)

        return self.__extract_hierarchy_for_chunks(relevant_chunks, observe=True)

    def __extract_hierarchy_for_chunks(self, relevant_chunks: [RelevantChunk], observe: bool = False) -> RetrievalOutput:
        retrieval_output_items = []
        for relevant_chunk in relevant_chunks:
            hierarchical_chunk_id = self.__chunk_id_for_hierarchy(relevant_chunk.chunk_id, self.max_levels)
            hierarchical_chunk = self.retriever.get_chunk(relevant_chunk.document_id, hierarchical_chunk_id)

            if observe:
                global_data["observer"].add_relevant_chunk(relevant_chunk.get_id(), hierarchical_chunk.chunk_text)


            relevant_item = RetrievalOutputItem(document_id=relevant_chunk.document_id,
                                                chunk_id=relevant_chunk.chunk_id,
                                                text=hierarchical_chunk.chunk_text)
            retrieval_output_items.append(relevant_item)

        return RetrievalOutput(retrieval_output_items)

    @staticmethod
    def __chunk_id_for_hierarchy(chunk_id: str, max_levels: int) -> [str]:
        """
        This method extracts the chunk id for the chunk max_levels higher in the hierarchy.

        chunk_id=doc1_1_3_2_1, max_levels=2 -> doc1_1_3
        chunk_id=doc1_1_3_2_1, max_levels=4 -> doc1_1

        :param chunk_id: The relevant chunk id
        :param max_levels: Maximum number of levels to go up in the hierarchy
        :return: A string with the found chunk_id
        """
        hierarchical_chunk_ids = chunk_id.split("_")

        # Calculate the number of levels we need to return, we need to subtract 1, we always keep the highest level.
        max_levels_to_go_up = min(max_levels, len(hierarchical_chunk_ids)-1)
        to_keep = len(hierarchical_chunk_ids) - max_levels_to_go_up

        return "_".join(hierarchical_chunk_ids[:to_keep])