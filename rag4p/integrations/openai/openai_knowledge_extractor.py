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
        prompt = f"""Task: Extract Knowledge Chunks

            Please extract knowledge chunks from the following text. Each chunk should capture distinct, self-contained 
            units of information in a subject-description format. Return the extracted knowledge chunks as a JSON 
            object or array, ensuring that each chunk includes both the subject and its corresponding description. 
            Typical subjects are: people, places, events, concepts, terms.
            Use the format: 
            {{"knowledge_chunks": [{{"subject": "subject", "description": "description"}}]}}

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
