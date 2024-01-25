import uuid

import weaviate


class AccessWeaviate:

    def __init__(self, url, access_key):
        print(f"Connecting to Weaviate at {url}")
        self.client = weaviate.connect_to_wcs(
            cluster_url=url,
            auth_credentials=weaviate.auth.AuthApiKey(access_key)
        )

    def does_collection_exist(self, collection_name):
        exists = self.client.collections.exists(collection_name)

        if not exists:
            print(f"Collection {collection_name} does not exist")

        return exists

    def add_document(self, collection_name: str,  properties: dict, vector: [float]):
        self.client.collections.get(collection_name).data.insert(
            uuid=uuid.uuid4(),
            properties=properties,
            vector=vector
        )

    def delete_collection(self, collection_name: str):
        self.client.collections.delete(collection_name)

    def create_collection(self, collection_name: str, properties: list):
        self.client.collections.create(
            name=collection_name,
            properties=properties
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
