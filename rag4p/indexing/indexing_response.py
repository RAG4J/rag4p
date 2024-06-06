from abc import ABC


class IndexingResponse(ABC):
    def __init__(self, num_documents: int, num_chunks: int, content_reader: str, splitter: str, running_time: float):
        self.num_documents = num_documents
        self.num_chunks = num_chunks
        self.content_reader = content_reader
        self.splitter = splitter
        self.running_time = running_time

    def __str__(self):
        return (f"IndexingResponse(num_documents={self.num_documents}, "
                f"num_chunks={self.num_chunks}, "
                f"content_reader={self.content_reader}, "
                f"splitter={self.splitter}, "
                f"running_time={self.running_time:.2f} sec.)")
