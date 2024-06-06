import uuid

import weaviate
import weaviate.classes as wvc


class AccessWeaviate:

    def __init__(self, url, access_key, openai_api_key: str = None):
        print(f"Connecting to Weaviate at {url}")
        headers = {}
        if openai_api_key:
            headers = {"X-OpenAI-Api-Key": openai_api_key}

        self.client = weaviate.connect_to_wcs(
            cluster_url=url,
            auth_credentials=weaviate.auth.AuthApiKey(access_key),
            headers=headers
        )

    def available_collections(self):
        collections = self.client.collections.list_all(simple=True)
        return [collection["name"] for collection in collections]

    def does_collection_exist(self, collection_name):
        exists = self.client.collections.exists(collection_name)

        if not exists:
            print(f"Collection {collection_name} does not exist")

        return exists

    def add_document(self, collection_name: str, properties: dict, vector: [float]):
        self.client.collections.get(collection_name).data.insert(
            uuid=uuid.uuid4(),
            properties=properties,
            vector=vector
        )

    def delete_collection(self, collection_name: str):
        self.client.collections.delete(collection_name)

    def create_collection(self, collection_name: str, properties: list, model: str = "text-embedding-3-small"):
        self.client.collections.create(
            name=collection_name,
            properties=properties,
            vectorizer_config=wvc.config.Configure.Vectorizer.text2vec_openai(
                model=model,
                type_="text",
            )
        )

    def force_create_collection(self, collection_name: str, properties: list):
        if self.does_collection_exist(collection_name):
            self.delete_collection(collection_name)

        self.create_collection(collection_name, properties)

    def close(self):
        self.client.close()

    def print_meta(self):
        meta = self.client.get_meta()
        print(f"Version: {meta['version']}")
        collections = self.client.collections.list_all(simple=False)
        for collection in collections:
            print(f"Available collection: {collection}")

    def obtain_meta(self):
        meta = self.client.get_meta()
        collections = self.client.collections.list_all(simple=False)
        meta["collections"] = collections
        return meta

    def obtain_meta_for_collection(self, collection_name: str):
        meta = self.client.get_meta()
        if self.does_collection_exist(collection_name):
            meta["collection"] = self.client.collections.export_config(name=collection_name)
        return meta

    def query_collection(self, question: str, collection_name: str, max_results: int = 2):
        print(f"Query collection based on user input {question}")

        if not collection_name or not self.does_collection_exist(collection_name):
            print(f"Collection {collection_name} does not exist")
            raise Exception(f"Collection {collection_name} does not exist")

        collection = self.client.collections.get(name=collection_name)
        return collection.query.hybrid(query=question,
                                       limit=max_results,
                                       alpha=0.5,
                                       fusion_type=wvc.query.HybridFusion.RELATIVE_SCORE,
                                       return_metadata=wvc.query.MetadataQuery(
                                           distance=True, score=True)
                                       )
