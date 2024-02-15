from rag4p.integrations.openai.openai_embedder import OpenAIEmbedder
from rag4p.integrations.weaviate import chunk_collection, CLASS_NAME
from rag4p.integrations.weaviate.access_weaviate import AccessWeaviate
from rag4p.integrations.weaviate.weaviate_content_store import WeaviateContentStore
from rag4p.indexing.splitters.sentence_splitter import SentenceSplitter
from rag4p.indexing.indexing_service import IndexingService
from rag4p.util.key_loader import KeyLoader
from rag4p.vasa_content_reader import VasaContentReader

import weaviate.classes.config as wvc

if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()

    key_loader = KeyLoader()
    access_weaviate = AccessWeaviate(url=key_loader.get_weaviate_url(), access_key=key_loader.get_weaviate_api_key())
    access_weaviate.force_create_collection(collection_name=CLASS_NAME,
                                            properties=chunk_collection.weaviate_properties(
                                                additional_properties=[
                                                    wvc.Property(name="title",
                                                                 data_type=wvc.DataType.TEXT,
                                                                 vectorize_property_name=False,
                                                                 skip_vectorization=True),
                                                    wvc.Property(name="timerange",
                                                                 data_type=wvc.DataType.TEXT,
                                                                 vectorize_property_name=False,
                                                                 skip_vectorization=True),
                                                ]
                                            )
                                        )

    embedder = OpenAIEmbedder(api_key=key_loader.get_openai_api_key())
    content_store = WeaviateContentStore(weaviate_access=access_weaviate,embedder=embedder)
    splitter = SentenceSplitter()
    indexing_service = IndexingService(content_store=content_store)

    content_reader = VasaContentReader()
    indexing_service.index_documents(content_reader=content_reader, splitter=splitter)

    access_weaviate.close()
