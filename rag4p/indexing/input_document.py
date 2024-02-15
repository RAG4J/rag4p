class InputDocument:
    document_id: str
    text: str
    properties: {}

    def __init__(self, document_id, text, properties):
        self.document_id = document_id
        self.text = text
        self.properties = properties
