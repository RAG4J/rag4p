from rag4p.indexing.input_document import InputDocument
from rag4p.indexing.jsonl_content_reader import JsonlContentReader


class VasaContentReader(JsonlContentReader):
    def __init__(self):
        super().__init__(file_path="../data/vasa-timeline.jsonl")

    def map_to_input_document(self, data) -> InputDocument:
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
        return document
