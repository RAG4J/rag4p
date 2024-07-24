import json
import re
from typing import List

from openai import OpenAI

from rag4p.integrations.ollama import DEFAULT_MODEL
from rag4p.integrations.ollama.access_ollama import AccessOllama
from rag4p.rag.generation.knowledge.knowledge import Knowledge
from rag4p.rag.generation.knowledge.knowledge_extractor import KnowledgeExtractor


class OllamaKnowledgeExtractor(KnowledgeExtractor):
    def __init__(self, access_ollama: AccessOllama, model: str = DEFAULT_MODEL):
        self.ollama = access_ollama
        self.model = model

    def extract_knowledge(self, context: str) -> List[Knowledge]:
        prompt = f"""Task: Extract Knowledge Chunks

            Please extract knowledge chunks from the following text. Each chunk should capture distinct, self-contained 
            units of information in a subject-description format. Return the extracted knowledge chunks as a JSON 
            object or array, ensuring that each chunk includes both the subject and its corresponding description. 
            Use the format: 
            {{"knowledge_chunks": [{{"subject": "subject", "description": "description"}}]}}

            Text:
            {context}
            """
        response = self.ollama.generate_answer(prompt=prompt, model=self.model)

        try:
            answer = json.loads(response)
        except json.JSONDecodeError:
            print(f"Error: Could not decode response from Ollama: {response}")
            return []

        k_items = []
        for kc in answer["knowledge_chunks"]:
            if "subject" not in kc or "description" not in kc:
                print(f"Error: Knowledge chunk missing subject or description: {kc}")
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
