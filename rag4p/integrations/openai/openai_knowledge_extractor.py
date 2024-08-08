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

Objective: Extract meaningful knowledge chunks from the provided text. Each chunk should be a distinct, self-contained unit of information presented in a subject-description format. It is essential that the information is from the input text. Do not make any assumptions or add any external information.

Instructions:
1. Identify distinct, relevant pieces of information from the text.
2. Ensure each piece of information focuses on one specific aspect: a person, an event, a location, an activity, a product, a concept, or a term.
3. Consolidate related pieces of information into broader categories only if they contribute to a clearer understanding of the subject.
4. For each piece of information, extract it as a "subject" and provide a corresponding detailed "description" taken from the input text. Do not include your own interpretation or additional information.
5. Ensure that the extracted chunks are formatted as a JSON object or array.
6. Provide enough context in the description to make each knowledge chunk understandable on its own.

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
