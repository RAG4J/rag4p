from abc import ABC, abstractmethod
from typing import List


class IndexComponent(ABC):
    def __init__(self, name: str, version: int):
        self.name = name
        self.version = version

    @abstractmethod
    def get_body(self):
        pass


class ComponentSettings(IndexComponent):
    def __init__(self, name: str, version: int, settings: dict):
        super().__init__(name, version)
        self.settings = {
            "template": {
                "settings": settings
            },
            "version": version,
        }

    def get_body(self):
        return self.settings


class ComponentMappings(IndexComponent):
    def __init__(self, name: str, version: int, mappings: dict):
        super().__init__(name, version)
        self.mappings = {
            "template": {
                "mappings": {
                    "properties": mappings
                }
            },
            "version": version,
        }

    def get_body(self):
        return self.mappings


class ComponentDynamicMappings(IndexComponent):
    def __init__(self, name: str, version: int, dynamic_mappings: List[dict]):
        super().__init__(name, version)
        if not dynamic_mappings or len(dynamic_mappings) == 0:
            dynamic_mappings = [{
                "strings": {
                    "match_mapping_type": "string",
                    "mapping": {
                        "type": "text",
                        "fields": {
                            "keyword": {
                                "type": "keyword",
                                "ignore_above": 256
                            }
                        }
                    }
                }
            }]

        self.dynamic_mappings = {
            "template": {
                "mappings": {
                    "dynamic_templates": dynamic_mappings
                }
            },
            "version": version,
        }

    def get_body(self):
        return self.dynamic_mappings


class ComponentTemplate(IndexComponent):
    def __init__(self, name: str, version: int, index_name: str, component_names: list):
        super().__init__(name, version)
        self.template = {
            "index_patterns": [f"{index_name}-*"],
            "priority": 10,
            "template": {
                "mappings": {
                    "properties": {
                        "document_id": {
                            "type": "keyword"
                        },
                        "chunk_id": {
                            "type": "integer"
                        },
                        "total_chunks": {
                            "type": "integer"
                        },
                        "chunk_text": {
                            "type": "text"
                        }
                    }
                },
                "settings": {
                    "number_of_shards": 1,
                    "number_of_replicas": 0,
                    "index": {
                        "knn": "true",
                        "knn.algo_param.ef_search": 8,
                    }
                }
            },
            "version": version,
            "composed_of": component_names
        }

    def get_body(self):
        return self.template
