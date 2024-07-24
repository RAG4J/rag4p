import json
from typing import List

from openai import OpenAI

from rag4p.integrations.openai import DEFAULT_MODEL
from rag4p.rag.generation.knowledge.knowledge import Knowledge
from rag4p.rag.generation.knowledge.knowledge_extractor import KnowledgeExtractor


class OpenaiKnowledgeExtractor(KnowledgeExtractor):
    def __init__(self, openai_api_key: str, openai_model: str = DEFAULT_MODEL):
        self.openai_client = OpenAI(
            api_key=openai_api_key,
        )
        self.openai_model = openai_model

    def extract_knowledge(self, context: str) -> List[Knowledge]:
        prompt = f"""
        Task: Extract Knowledge Chunks

        Objective: Extract knowledge chunks from the provided text. Each chunk should be a distinct, self-contained unit of information presented in a subject-description format.

        Instructions:
        1. Identify key pieces of information from the text.
        2. Consolidate related pieces of information into broader categories where possible.
        3. For each consolidated piece of information, extract it as a "subject" and provide a corresponding "description."
        4. Ensure that the extracted chunks are formatted as a JSON object or array.
        5. Typical subjects include: people, places, events, concepts, terms. Broaden the subjects to avoid overly narrow categories.

        Format:
        {{
            "knowledge_chunks": [
                {{"subject": "subject", "description": "description"}},
                ...
            ]
        }}

        Text:
        {context}
        """
        completion = self.openai_client.chat.completions.create(
            model=self.openai_model,
            response_format={"type": "json_object"},
            messages=[
                {
                    "role": "system",
                    "content": "You are an assistant that takes apart a piece of text into semantic chunks to be used "
                               "in a RAG system."},
                {"role": "user", "content": prompt},
            ],
            stream=False,
        )

        answer = json.loads(completion.choices[0].message.content)

        k_items = []
        for kc in answer["knowledge_chunks"]:
            item = Knowledge(kc["subject"], kc["description"])
            k_items.append(item)

        return k_items
