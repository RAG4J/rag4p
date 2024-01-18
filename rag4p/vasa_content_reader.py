import json

from rag4p.domain.input_document import InputDocument
from rag4p.indexing.content_reader import ContentReader


class VasaContentReader(ContentReader):
    def __init__(self):
        self.filename = "../data/vasa-timeline.jsonl"

    def read(self):
        with open(self.filename, 'r') as file:
            for line in file:
                data = json.loads(line)
                properties = {
                    "title": data["title"],
                    "timerange": data["timerange"]
                }
                document_id = data["title"].lower().replace(" ", "-")
                document = InputDocument(
                    document_id=document_id,
                    text=data["body"],
                    properties=properties
                )
                yield document
