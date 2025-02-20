from typing import List

from rag4p.indexing.indexing_service import IndexingService
from rag4p.integrations.openai import MODEL_GPT4_TURBO
from rag4p.integrations.openai.openai_embedder import OpenAIEmbedder
from rag4p.integrations.openai.openai_question_generator import OpenAIQuestionGenerator
from rag4p.indexing.input_document import InputDocument
from rag4p.indexing.splitters.sentence_splitter import SentenceSplitter
from rag4p.logging_config import setup_logging
from rag4p.rag.generation.question_generator_service import QuestionGeneratorService
from rag4p.rag.retrieval.quality.retrieval_quality_service import obtain_retrieval_quality
from rag4p.rag.store.local.internal_content_store import InternalContentStore

from rag4p.rag.retrieval.quality.question_answer_record import QuestionAnswerRecord
from rag4p.util.key_loader import KeyLoader

if __name__ == '__main__':
    from dotenv import load_dotenv

    load_dotenv()
    setup_logging()

    key_loader = KeyLoader()
    embedder = OpenAIEmbedder(api_key=key_loader.get_openai_api_key())
    content_store = InternalContentStore(embedder=embedder)
    doc_text = ("The Vasa was a Swedish warship built between 1626 and 1628. The ship foundered and sank after sailing "
                "about 1,300 m (1,400 yd) into its maiden voyage on 10 August 1628.")
    indexing_service = IndexingService(content_store=content_store)
    indexing_service.index_document(InputDocument(document_id="vasa", text=doc_text, properties={}),
                                    splitter=SentenceSplitter())

    question_generator = OpenAIQuestionGenerator(openai_api_key=key_loader.get_openai_api_key(),
                                                 openai_model=MODEL_GPT4_TURBO)
    question_generator_service = QuestionGeneratorService(retriever=content_store,
                                                          question_generator=question_generator)

    question_answer_pairs = []  # type: List[QuestionAnswerRecord]
    for chunk in content_store.loop_over_chunks():
        # Use LLM to generate a question for this chunk
        question = question_generator_service.generate_question(chunk.chunk_text)

        question_answer_pairs.append(QuestionAnswerRecord(document_id=chunk.document_id,
                                                          chunk_id=chunk.chunk_id,
                                                          chunk_text=chunk.chunk_text,
                                                          question=question))

    retriever_quality = obtain_retrieval_quality(question_answer_records=question_answer_pairs, retriever=content_store)

    print(f"Quality using precision: {retriever_quality.precision()}")
    print(f"Total questions: {retriever_quality.total_questions()}")
