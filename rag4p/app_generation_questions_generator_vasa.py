from rag4p.integrations.openai import MODEL_GPT4_TURBO
from rag4p.integrations.openai.openai_question_generator import OpenAIQuestionGenerator
from rag4p.indexing.splitters.sentence_splitter import SentenceSplitter
from rag4p.indexing.indexing_service import IndexingService
from rag4p.rag.embedding.local.onnx_embedder import OnnxEmbedder
from rag4p.rag.generation.question_generator_service import QuestionGeneratorService
from rag4p.rag.store.local.internal_content_store import InternalContentStore
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

    question_generator = OpenAIQuestionGenerator(openai_api_key=key_loader.get_openai_api_key(),
                                                 openai_model=MODEL_GPT4_TURBO)

    question_generator_service = QuestionGeneratorService(retriever=content_store,
                                                          question_generator=question_generator)

    question_generator_service.generate_question_answer_pairs(file_name="questions_answers.csv")
