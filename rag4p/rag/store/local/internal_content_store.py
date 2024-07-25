from datetime import datetime
import json
import pickle
from typing import List

import pandas as pd

from rag4p.rag.model.chunk import Chunk
from rag4p.rag.model.relevant_chunk import RelevantChunk
from rag4p.rag.store.content_store import ContentStore
from rag4p.rag.embedding.embedder import Embedder
from rag4p.rag.retrieval.retriever import Retriever

from scipy.spatial import distance


class InternalContentStore(ContentStore, Retriever):
    """
    The internal content stores stores the chunks in memory, it acts as a normal content store, but it als contains
    all the methods from a retriever, so it can be used as a retriever as well. This is useful for testing purposes.
    """

    def __init__(self, embedder: Embedder, metadata=None):
        _metadata = {
                'name': 'internal-content-store',
                'create_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'embedder': embedder.identifier(),
                'supplier': embedder.supplier(),
                'model': embedder.model()
            }

        if metadata is not None:
            _metadata = {**_metadata, **metadata}

        super().__init__(_metadata)
        self.embedder = embedder
        self.vector_store = pd.DataFrame(columns=['chunk_id', 'chunk', 'embedding'])

    def store(self, chunks: List[Chunk]):
        for chunk in chunks:
            self.__store_chunk(chunk)

    def __store_chunk(self, chunk: Chunk):
        # Check if chunk.chunk_text has content other than whitespace
        if not chunk.chunk_text.strip():
            print(f"Chunk {chunk.chunk_id}: '{chunk.chunk_text}' has no content")
            return
        chunk_id = chunk.document_id + "_" + str(chunk.chunk_id)
        print(f"Storing chunk {chunk_id}: {chunk.chunk_text}")
        try:
            embedding = self.embedder.embed(chunk.chunk_text)
            self.vector_store.loc[len(self.vector_store)] = {'chunk_id': chunk_id, 'chunk': chunk, 'embedding': embedding}
        except Exception as e:
            print(f"Error storing chunk {chunk_id}-{chunk.chunk_text}: {e}")

    def find_relevant_chunks(self, query: str, max_results: int = 4) -> List[RelevantChunk]:
        print(f"Finding relevant chunks for query: {query}")
        embedding = self.embedder.embed(query)
        self.vector_store['distance'] = self.vector_store['embedding'].apply(lambda x: distance.euclidean(x, embedding))
        relevant_chunks_df = self.vector_store.nsmallest(max_results, 'distance')

        relevant_chunks = []
        for index, row in relevant_chunks_df.iterrows():
            chunk = row['chunk']
            score = row['distance']
            relevant_chunk = RelevantChunk(
                document_id=chunk.document_id,
                chunk_id=chunk.chunk_id,
                total_chunks=chunk.total_chunks,
                text=chunk.chunk_text,
                properties=chunk.properties,
                score=score
            )
            relevant_chunks.append(relevant_chunk)
        return relevant_chunks

    def get_chunk_by_id(self, chunk_id: str) -> Chunk:
        """
        Obtains a chunk using its complete id (document_id + "_" + chunk_id)
        :param chunk_id: complete id of the chunk document_id + "_" + chunk_id
        :return:
        """
        found_chunk = self.vector_store[self.vector_store['chunk_id'] == chunk_id]

        if len(found_chunk) == 0:
            raise Exception(f"Chunk with id {chunk_id} not found.")

        return found_chunk.iloc[0]['chunk']

    def loop_over_chunks(self):
        for index, row in self.vector_store.iterrows():
            yield row['chunk']

    def backup(self, path: str):
        # Save the DataFrame to a pickle file
        with open(f'{path}.pickle', 'wb') as f:
            pickle.dump(self.vector_store, f)

        # Save the metadata to a JSON file
        with open(f'{path}_metadata.json', 'w') as f:
            json.dump(self._metadata, f)

    @classmethod
    def load_from_backup(cls, embedder: Embedder, path: str):
        # Create an instance of the class
        instance = cls(embedder)

        # Load the DataFrame from the pickle file
        with open(f'{path}.pickle', 'rb') as f:
            instance.vector_store = pickle.load(f)

        # Load the metadata from the JSON file
        with open(f'{path}_metadata.json', 'r') as f:
            instance._metadata = json.load(f)

        if 'embedder' in instance._metadata:
            if instance._metadata['embedder'] != embedder.identifier():
                raise Exception(f"Embedder {embedder.identifier()} does not match the one in the backup: "
                                f"{instance._metadata['embedder']}")

        return instance
