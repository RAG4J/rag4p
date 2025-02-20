from rag4p.integrations.weaviate.access_weaviate import AccessWeaviate
from rag4p.logging_config import setup_logging
from rag4p.util.key_loader import KeyLoader

if __name__ == '__main__':
    from dotenv import load_dotenv
    load_dotenv()
    setup_logging()

    key_loader = KeyLoader()
    weaviate_client = AccessWeaviate(url=key_loader.get_weaviate_url(), access_key=key_loader.get_weaviate_api_key())
    try:
        weaviate_client.print_meta()
    finally:
        weaviate_client.close()
