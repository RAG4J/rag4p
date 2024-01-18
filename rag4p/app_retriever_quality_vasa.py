from rag4p.indexing.SentenceSplitter import SentenceSplitter
from rag4p.indexing.indexing_service import IndexingService
from rag4p.onnxembedder.onnx_embedder import OnnxEmbedder
from rag4p.quality.retrieval_quality_service import read_question_answers_from_file, obtain_retrieval_quality
from rag4p.store.internal_content_retriever import InternalContentRetriever
from rag4p.store.internal_content_store import InternalContentStore
from rag4p.util.key_loader import KeyLoader
from rag4p.vasa_content_reader import VasaContentReader

if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()

    key_loader = KeyLoader()
    embedder = OnnxEmbedder()
    content_store = InternalContentStore(embedder=embedder)
    indexing_service = IndexingService(content_store=content_store)
    indexing_service.index_documents(content_reader=VasaContentReader(), splitter=SentenceSplitter())
    retriever = InternalContentRetriever(internal_content_store=content_store)

    question_answer_records = read_question_answers_from_file()
    retriever_quality = obtain_retrieval_quality(question_answer_records=question_answer_records, retriever=retriever)

    print(f"Quality using precision: {retriever_quality.precision()}")
    print(f"Total questions: {retriever_quality.total_questions()}")
