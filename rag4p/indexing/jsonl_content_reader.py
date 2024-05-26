import json
from abc import abstractmethod
from typing import Iterable, List

from rag4p.indexing.content_reader import ContentReader
from rag4p.indexing.input_document import InputDocument


class JsonlContentReader(ContentReader):
    def __init__(self, file_path: str):
        self.file_path = file_path

    def read(self, batch_size: int = 10) -> Iterable[List[InputDocument]]:
        with open(self.file_path, 'r') as file:
            batch = []
            for line in file:
                data = json.loads(line)
                document = self.map_to_input_document(data)
                batch.append(document)
                if len(batch) == batch_size:
                    yield batch
                    batch = []
            if batch:
                yield batch

    @abstractmethod
    def map_to_input_document(self, data) -> InputDocument:
        pass
