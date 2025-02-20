import logging

from rag4p.integrations.opensearch.index_components import ComponentTemplate, ComponentSettings, \
    ComponentDynamicMappings, ComponentMappings, IndexComponent
from rag4p.integrations.opensearch.opensearch_client import OpenSearchClient

ost_logger = logging.getLogger(__name__)

class OpenSearchTemplate:

    def __init__(self,
                 client: OpenSearchClient,
                 index_template: ComponentTemplate,
                 component_settings: ComponentSettings,
                 component_dyn_mappings: ComponentDynamicMappings,
                 component_mappings: ComponentMappings):
        self.client = client
        self.index_template = index_template
        self.component_settings = component_settings
        self.component_dyn_mappings = component_dyn_mappings
        self.component_mappings = component_mappings

    def create_update_template(self) -> list:
        """ Check the version of the current template and update if necessary. """
        ost_logger.info("Initialize or update the product template in OpenSearch.")

        return [
            self.__update_component(component=self.component_settings),
            self.__update_component(component=self.component_dyn_mappings),
            self.__update_component(component=self.component_mappings),
            self.__update_index_template(template=self.index_template)
        ]

    def __update_component(self, component: IndexComponent):
        body = component.get_body()
        required_version = component.version
        name = component.name
        component_needs_update = self.__component_needs_update(component_name=name,
                                                               current_version=required_version)
        if component_needs_update:
            self.client.set_component_template(name=name, body=body)
            result = f"Update the component template {name} to version {required_version}."
        else:
            result = f"The version {required_version} of the component template {name} is up-to-date"

        return result

    def __update_index_template(self, template: ComponentTemplate):
        body = template.get_body()
        required_version = template.version
        name = template.name
        template_needs_update = self.__template_needs_update(template_name=name,
                                                             current_version=required_version)

        if template_needs_update:
            self.client.set_index_template(name=name, body=body)
            result = f"Update the template to version {required_version}."
        else:
            result = f"The version {required_version} of the index template is up-to-date"

        return result

    def __template_needs_update(self, template_name: str, current_version):
        """ The template needs to update if there is no template or if the versions do not match """
        if not self.client.does_index_template_exist(name=template_name):
            ost_logger.info("The template with name '%s' is not found", template_name)
            return True

        response = self.client.get_index_template(name=template_name)
        ost_logger.debug(response)

        templates = response['index_templates']
        if len(templates) != 1:
            raise Exception(f"We cannot have matching more than 1 template while looking for {template_name}")

        index_template = templates[0]['index_template']
        return not (index_template.get("version") == current_version)

    def __component_needs_update(self, component_name: str, current_version):
        """ The template needs to update if there is no template or if the versions do not match """
        ost_logger.info("Obtain the component template with name '%s'", component_name)
        if not self.client.does_component_template_exist(name=component_name):
            ost_logger.info(f"The component template with name '%s' is not found", component_name)
            return True

        response = self.client.get_component_template(name=component_name)
        ost_logger.debug(response)

        components = response['component_templates']
        if len(components) != 1:
            raise Exception("We cannot have matching more than 1 component while looking for " + component_name)

        component_template = components[0]['component_template']
        return not (component_template.get("version") == current_version)
