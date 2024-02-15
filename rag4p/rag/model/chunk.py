class Chunk:
    document_id: str
    chunk_id: int
    total_chunks: int
    chunk_text: str
    properties: {}

    def __init__(self, document_id: str, chunk_id: int, total_chunks: int, chunk_text: str, properties: dict):
        self.document_id = document_id
        self.chunk_id = chunk_id
        self.total_chunks = total_chunks
        self.chunk_text = chunk_text
        self.properties = properties

    def get_id(self):
        return self.document_id + "_" + str(self.chunk_id)
