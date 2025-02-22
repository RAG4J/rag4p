from rag4p.indexing.splitters.sentence_splitter import SentenceSplitter
from rag4p.indexing.indexing_service import IndexingService
from rag4p.integrations.ollama.access_ollama import AccessOllama
from rag4p.integrations.ollama.ollama_embedder import OllamaEmbedder
from rag4p.logging_config import setup_logging
from rag4p.rag.retrieval.quality.retrieval_quality_service import read_question_answers_from_file, \
    obtain_retrieval_quality
from rag4p.rag.store.local.internal_content_store import InternalContentStore
from rag4p.util.key_loader import KeyLoader
from rag4p.vasa_content_reader import VasaContentReader

if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    setup_logging()

    key_loader = KeyLoader()
    access_ollama = AccessOllama()
    embedder = OllamaEmbedder(access_ollama=access_ollama)
    content_store = InternalContentStore(embedder=embedder)
    indexing_service = IndexingService(content_store=content_store)
    indexing_service.index_documents(content_reader=VasaContentReader(), splitter=SentenceSplitter())

    question_answer_records = read_question_answers_from_file()
    retriever_quality = obtain_retrieval_quality(question_answer_records=question_answer_records,
                                                 retriever=content_store)

    print(f"Quality using precision: {retriever_quality.precision()}")
    print(f"Total questions: {retriever_quality.total_questions()}")
