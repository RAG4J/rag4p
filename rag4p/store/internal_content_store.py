from typing import List

import pandas as pd

from rag4p.domain.chunk import Chunk
from rag4p.indexing.content_store import ContentStore
from rag4p.indexing.embedder import Embedder

import numpy as np
from scipy.spatial import distance


class InternalContentStore(ContentStore):
    """
    The internal content stores stores the chunks in memory, it acts as a normal content store, but it als contains
    all the methods from a retriever, so it can be used as a retriever as well. This is useful for testing purposes.
    """
    def __init__(self, embedder: Embedder):
        super().__init__(embedder)
        self.df = pd.DataFrame(columns=['chunk_id', 'chunk', 'embedding'])

    def store(self, chunks: List[Chunk]):
        for chunk in chunks:
            self.__store_chunk(chunk)

    def __store_chunk(self, chunk: Chunk):
        chunk_id = chunk.document_id + "_" + str(chunk.chunk_id)
        print(f"Storing chunk {chunk_id}: {chunk.chunk_text}")
        embedding = self.embedder.embed(chunk.chunk_text)
        self.df.loc[len(self.df)] = {'chunk_id': chunk_id, 'chunk': chunk, 'embedding': embedding}

    def find_relevant_chunks(self, query: str, max_items: int = 4) -> List[Chunk]:
        print(f"Finding relevant chunks for query: {query}")
        embedding = self.embedder.embed(query)
        self.df['distance'] = self.df['embedding'].apply(lambda x: distance.euclidean(x, embedding))
        relevant_chunks_df = self.df.nsmallest(max_items, 'distance')
        return relevant_chunks_df['chunk'].tolist()

    def get_chunk(self, chunk_id: str) -> Chunk:
        """
        Obtains a chunk using its complete id (document_id + "_" + chunk_id)
        :param chunk_id: complete id of the chunk document_id + "_" + chunk_id
        :return:
        """
        found_chunk = self.df[self.df['chunk_id'] == chunk_id]

        if len(found_chunk) == 0:
            raise Exception(f"Chunk with id {chunk_id} not found.")

        return found_chunk.iloc[0]['chunk']

    def loop_over_chunks(self) -> None:
        for index, row in self.df.iterrows():
            yield row['chunk']