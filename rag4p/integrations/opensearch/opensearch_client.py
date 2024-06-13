import logging
from datetime import datetime

from opensearchpy import OpenSearch, RequestsHttpConnection

search_log = logging.getLogger("search")


class OpenSearchClient:
    def __init__(self, client: OpenSearch, alias_name: str = None):
        self.opensearch = client

        if alias_name:
            self.default_alias_name = alias_name

    def ping(self):
        if self.opensearch.ping():
            print('Connected to OpenSearch')
            return True
        else:
            print('Could not connect to OpenSearch')
            return False

    def create_index(self, provided_alias_name: str = None):
        """
        Create a new index. Name of the index is a combination of the provided or default alias name and a time stamp
        in the format of YearMonthDayHourMinuteSecond. Before the index is created, we remove it if it already exists.
        The settings and mappings are obtained from the shoes_index.json in the config folder.
        :return: The name of the created index
        """
        alias_name = self.__get_alias_name(provided_alias_name)
        index_name = f'{alias_name}-{datetime.now().strftime("%Y%m%d%H%M%S")}'

        self.opensearch.indices.delete(index=index_name, ignore_unavailable=True)
        self.opensearch.indices.create(index=index_name)

        print(f'Created a new index with the name {index_name}')
        return index_name

    def switch_alias_to(self, index_name: str, provided_alias_name: str = None):
        """
        Checks if the alias as configured is already available, if so, remove all indexes it points to. When finished add
        the provided index to the alias.
        :param provided_alias_name: Overrides the default_alias_name.
        :param index_name: Name of the index to assign to the alias
        :return:
        """
        alias_name = self.__get_alias_name(provided_alias_name)
        print(f'Assign alias {alias_name} to {index_name}')
        body = {
            "actions": [
                {"remove": {"index": f'{alias_name}-*', "alias": alias_name}},
                {"add": {"index": index_name, "alias": alias_name}}
            ]
        }
        self.opensearch.indices.update_aliases(body=body)

    def index_document(self, id: str, document: dict, index_name: str):
        """
        Send the provided shoe to Elasticsearch to index that shoe into the provided index.
        :param id: Identifier to use for the index
        :param document: The document to index
        :param index_name: The index to use for indexing the shoe
        :return:
        """
        print(f'Indexing item: {id} into index with name {index_name}')
        self.opensearch.index(index=index_name, id=id, body=document)

    def search(self, body, explain: bool = False, size: int = 10, index_name: str = None):
        index_or_alias = self.__get_alias_name(index_name)
        search_results = self.opensearch.search(index=index_or_alias, body=body, explain=explain, size=size)
        return search_results

    def count_docs(self, index_name: str = None):
        req_index = index_name if index_name is not None else self.default_alias_name
        return self.opensearch.count(index=req_index)

    def set_component_template(self, name, body):
        self.opensearch.cluster.put_component_template(name=name, body=body)

    def set_index_template(self, name, body):
        self.opensearch.indices.put_index_template(name=name, body=body)

    def does_index_template_exist(self, name: str):
        return self.opensearch.indices.exists_index_template(name=name)

    def get_index_template(self, name: str):
        return self.opensearch.indices.get_index_template(name=name)

    def does_component_template_exist(self, name: str):
        return self.opensearch.cluster.exists_component_template(name=name)

    def get_component_template(self, name: str):
        return self.opensearch.cluster.get_component_template(name=name)

    def delete_index(self, index_name: str):
        self.opensearch.indices.delete(index=index_name, ignore_unavailable=True)

    def client(self) -> OpenSearch:
        return self.opensearch

    def __get_alias_name(self, provided_alias_name: str = None) -> str:
        alias_name = provided_alias_name if provided_alias_name is not None else self.default_alias_name

        if not alias_name:
            print("We mandate using aliases for an index. Provided the alias while construction the client"
                            "or provided it with the appropriate function call")
            raise ValueError("We mandate using aliases for an index. Provided the alias while construction the client"
                             "or provided it with the appropriate function call")

        return alias_name
