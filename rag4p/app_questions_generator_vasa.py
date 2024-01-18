from rag4p.indexing.SentenceSplitter import SentenceSplitter
from rag4p.indexing.indexing_service import IndexingService
from rag4p.onnxembedder.onnx_embedder import OnnxEmbedder
from rag4p.quality.question_generator_service import QuestionGeneratorService
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

    question_generator_service = QuestionGeneratorService(openai_key=key_loader.get_openai_api_key(),
                                                          retriever=retriever)

    question_generator_service.generate_question_answer_pairs(file_name="questions_answers.csv")

