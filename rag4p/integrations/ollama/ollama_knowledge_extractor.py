import json
import logging
import re
from typing import List

from rag4p.integrations.ollama import DEFAULT_MODEL
from rag4p.integrations.ollama.access_ollama import AccessOllama
from rag4p.rag.generation.knowledge.knowledge import Knowledge
from rag4p.rag.generation.knowledge.knowledge_extractor import KnowledgeExtractor

oke_logger = logging.getLogger("oke_logger")

class OllamaKnowledgeExtractor(KnowledgeExtractor):
    def __init__(self, access_ollama: AccessOllama, model: str = DEFAULT_MODEL):
        self.ollama = access_ollama
        self.model = model

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
        response = self.ollama.generate_answer(prompt=prompt, model=self.model)

        try:
            answer = json.loads(response)
        except json.JSONDecodeError:
            oke_logger.exception(f"Error decoding response from Ollama in OllamaKnowledgeExtractor.extract_knowledge.")
            return []

        k_items = []
        for kc in answer["knowledge_chunks"]:
            if "subject" not in kc or "description" not in kc:
                oke_logger.error(f"Error: Knowledge chunk missing subject or description: {kc}")
                continue
            item = Knowledge(kc["subject"], kc["description"])
            k_items.append(item)

        return k_items


def clean_json_string(s):
    # Trim leading and trailing whitespace
    s = s.strip()
    # Escape control characters
    s = s.encode('unicode_escape').decode('utf-8')
    # Remove trailing commas from arrays and objects
    s = re.sub(r',\s*}', '}', s)
    s = re.sub(r',\s*]', ']', s)
    return s
