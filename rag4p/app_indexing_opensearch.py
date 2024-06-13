from dotenv import load_dotenv

from rag4p.indexing.indexing_service import IndexingService
from rag4p.indexing.splitters.sentence_splitter import SentenceSplitter
from rag4p.integrations.openai.openai_embedder import OpenAIEmbedder
from rag4p.integrations.opensearch.connection_builder import build_aws_search_service
from rag4p.integrations.opensearch.index_components import ComponentTemplate, ComponentSettings, \
    ComponentDynamicMappings, ComponentMappings
from rag4p.integrations.opensearch.opensearch_client import OpenSearchClient
from rag4p.integrations.opensearch.opensearch_content_store import OpenSearchContentStore
from rag4p.integrations.opensearch.opensearch_template import OpenSearchTemplate
from rag4p.util.key_loader import KeyLoader
from rag4p.vasa_content_reader import VasaContentReader

if __name__ == '__main__':
    load_dotenv()
    key_loader = KeyLoader()

    opensearch_client = build_aws_search_service()
    opensearch_client = OpenSearchClient(opensearch_client)

    index_template = ComponentTemplate(name="vasa_index_template",
                                       version=1,
                                       index_name="rag4p-vasa",
                                       component_names=["vasa_settings", "vasa_mappings", "vasa_dynamic_mappings"])
    component_settings = ComponentSettings(name="vasa_settings", version=1, settings={})
    component_dyn_mappings = ComponentDynamicMappings(name="vasa_dynamic_mappings", version=1, dynamic_mappings=[])
    component_mappings = ComponentMappings(name="vasa_mappings", version=1, mappings={
        "title": {
            "type": "text"
        },
        "timerange": {
            "type": "text"
        },
    })
    opensearch_template = OpenSearchTemplate(client=opensearch_client,
                                             index_template=index_template,
                                             component_settings=component_settings,
                                             component_dyn_mappings=component_dyn_mappings,
                                             component_mappings=component_mappings)

    opensearch_template.create_update_template()
    index_name = opensearch_client.create_index(provided_alias_name="rag4p-vasa")

    embedder = OpenAIEmbedder(api_key=key_loader.get_openai_api_key())
    opensearch_content_store = OpenSearchContentStore(opensearch_client=opensearch_client,
                                                      embedder=embedder,
                                                      index_name=index_name)

    splitter = SentenceSplitter()
    indexing_service = IndexingService(content_store=opensearch_content_store)

    content_reader = VasaContentReader()
    indexing_service.index_documents(content_reader=content_reader, splitter=splitter)

    opensearch_client.switch_alias_to(index_name=index_name, provided_alias_name="rag4p-vasa")
