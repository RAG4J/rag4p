from rag4p.rag.retrieval.retrieval_strategy import RetrievalStrategy
from rag4p.rag.retrieval.retriever import Retriever
from rag4p.rag.retrieval.retrieval_output import RetrievalOutput, RetrievalOutputItem
from rag4p.rag.model.relevant_chunk import RelevantChunk
from rag4p.rag.tracker.rag_tracker import global_data


class DocumentRetrievalStrategy(RetrievalStrategy):
    """
    This retrieval strategy retrieves complete documents for the relevant chunks. This strategy is useful when you want
    to compare parts of a document to vectors, but want the complete document for the context of an LLM.

    Imagine you have blogs posts as documents and you want to asnwer questions about them. The body of the blog is
    splitted into chunks, now you want to answer who wrote about a specific topic. The authors of the blogs posts are
    in the additional properties of the chunks. You can use this strategy to retrieve the complete blog post and the
    author of the blog post. Using this context, the LLM can find the author of the specific found blog post.

    When the chunks are created using a chain of splitters, the complete document is retrieved by combining all chunks
    from the highest level of the chain.
    """

    def __init__(self, retriever: Retriever, observe: bool = False):
        self.retriever = retriever
        self.observe = observe

    def retrieve_max_results(self, question, max_results) -> RetrievalOutput:
        relevant_chunks = self.retriever.find_relevant_chunks(question, max_results)
        return self.__extract_document_for_chunk(relevant_chunks, self.observe)

    def retrieve_max_results_observed(self, question, max_results) -> RetrievalOutput:
        relevant_chunks = self.retriever.find_relevant_chunks(question, max_results)

        return self.__extract_document_for_chunk(relevant_chunks, observe=True)

    def __extract_document_for_chunk(self, relevant_chunks: [RelevantChunk], observe: bool = False) -> RetrievalOutput:
        retrieval_output_items = []

        # Remove chunks from the same document
        unique_docs = self.__unique_documents_from_chunks(relevant_chunks)

        for relevant_chunk in unique_docs:
            # for a chunk that is not of the first splitter level, we need to obtain the parent chunk that is of the
            # first splitter level
            if len(relevant_chunk.chunk_id.split("_")) > 1:
                relevant_chunk = self.retriever.get_chunk(relevant_chunk.document_id, relevant_chunk.chunk_id.split("_")[0])

            overall_text = self.__read_text_from_all_chunks_for_document(relevant_chunk.document_id,
                                                                         relevant_chunk.total_chunks)

            # A chunk can have additional properties, add them to the text as well in the format of key: value
            for key, value in relevant_chunk.properties.items():
                overall_text += f"\n{key}: {value} "

            relevant_item = RetrievalOutputItem(document_id=relevant_chunk.document_id,
                                                chunk_id=relevant_chunk.chunk_id,
                                                text=overall_text)
            retrieval_output_items.append(relevant_item)

            if observe:
                print(f"'{relevant_chunk.get_id()}'")
                print(f"'{overall_text}'")
                global_data["observer"].add_relevant_chunk(relevant_chunk.get_id(), overall_text)

        return RetrievalOutput(retrieval_output_items)

    def __read_text_from_all_chunks_for_document(self, document_id, total_chunks):
        """
        A document consists of multiple numbered chunks, we need to combine all chunks to get the complete document.
        :param document_id: Document to get all chunks for.
        :param total_chunks: Total amount of chunks in the document.
        :return: The complete text of all chunks combined.
        """
        chunk_ids = range(0, total_chunks)
        overall_text = ""
        for chunk_id in chunk_ids:
            chunk = self.retriever.get_chunk(document_id, str(chunk_id))
            overall_text += chunk.chunk_text + " "
        return overall_text

    @staticmethod
    def __unique_documents_from_chunks(relevant_chunks):
        """
        Remove chunks from the same document, as we are returning complete documents.
        :param relevant_chunks: Provided relevant chunks
        :return: Relevant chunks from unique documents only
        """
        unique_docs_dict = {chunk.document_id: chunk for chunk in relevant_chunks}
        return list(unique_docs_dict.values())
