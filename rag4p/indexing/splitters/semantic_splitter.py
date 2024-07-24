from typing import List

from rag4p.indexing.input_document import InputDocument
from rag4p.indexing.splitter import Splitter
from rag4p.rag.generation.knowledge.knowledge_extractor import KnowledgeExtractor
from rag4p.rag.model.chunk import Chunk


class SemanticSplitter(Splitter):

    def __init__(self, knowledge_extractor: KnowledgeExtractor):
        self.knowledge_extractor = knowledge_extractor

    def split(self, input_document: InputDocument, parent_chunk: Chunk = None) -> List[Chunk]:
        input_text = input_document.text if parent_chunk is None else parent_chunk.chunk_text
        knowledge_items = self.knowledge_extractor.extract_knowledge(input_text)

        chunks_ = []
        for i, knowledge_item in enumerate(knowledge_items):
            chunk_id = str(i) if parent_chunk is None else f"{parent_chunk.chunk_id}_{i}"
            chunk_ = Chunk(input_document.document_id,
                           chunk_id,
                           len(knowledge_items),
                           f"{knowledge_item.subject}: {knowledge_item.description}",
                           input_document.properties)
            chunks_.append(chunk_)

        return chunks_

    @staticmethod
    def name() -> str:
        return SemanticSplitter.__name__
