from typing import List


class RetrievalOutputItem:
    document_id: str
    chunk_id: int
    text: str

    def __init__(self, document_id, chunk_id, text):
        self.document_id = document_id
        self.chunk_id = chunk_id
        self.text = text

    pass


class RetrievalOutput:
    items: List[RetrievalOutputItem]

    def __init__(self, items):
        self.items = items

    def construct_context(self):
        return ' '.join(item.text for item in self.items)

