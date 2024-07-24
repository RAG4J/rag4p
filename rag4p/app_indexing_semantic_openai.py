from rag4p.indexing.input_document import InputDocument
from rag4p.indexing.splitter_chain import SplitterChain
from rag4p.indexing.splitters.section_splitter import SectionSplitter
from rag4p.indexing.splitters.semantic_splitter import SemanticSplitter
from rag4p.integrations.ollama import MODEL_GEMMA2, MODEL_PHI3
from rag4p.integrations.ollama.access_ollama import AccessOllama
from rag4p.integrations.ollama.ollama_knowledge_extractor import OllamaKnowledgeExtractor
from rag4p.integrations.openai.openai_embedder import OpenAIEmbedder
from rag4p.integrations.openai.openai_knowledge_extractor import OpenaiKnowledgeExtractor
from rag4p.util.key_loader import KeyLoader

input_text = f"""Ever thought about building your very own question-answering system? Like the one that powers Siri, 
Alexa, or Google Assistant? Well, we've got something awesome lined up for you! In our hands-on workshop, we'll guide 
you through the ins and outs of creating a question-answering system. We prefer using Python for the workshop. We 
have prepared a GUI that works with python. If you prefer another language, you can still do the workshop, 
but you will miss the GUI to test your application.

You'll get your hands dirty with vector stores and Large Language Models, we help you combine these two in a way 
you've never done before. You've probably used search engines for keyword-based searches, right? Well, prepare to 
have your mind blown. We'll dive into something called semantic search, which is the next big thing after traditional 
searches. It’s like moving from asking Google to search "best pizza places" to "Where can I find a pizza place that 
my gluten-intolerant, vegan friend would love?" – you get the idea, right?

We’ll be teaching you how to build an entire pipeline, starting from collecting data from various sources, 
converting that into vectors (yeah, it’s more math, but it’s cool, we promise), and storing it so you can use it to 
answer all sorts of queries. It's like building your own mini Google!

We've got a repository ready to help you set up everything you need on your laptop. By the end of our workshop, 
you'll have your question-answering system ready and running. So, why wait? Grab your laptop, bring your coding hat, 
and let's start building something fantastic together. Trust us, it’s going to be a blast!

Some of the highlights of the workshop: 
- Use a vector store (OpenSearch, Elasticsearch, Weaviate)
- Use a Large Language Model (OpenAI, HuggingFace, Cohere, PaLM, Bedrock)
- Use a tool for content extraction (Unstructured, Llama)
- Create your pipeline (Langchain, Custom)
"""

if __name__ == '__main__':
    from dotenv import load_dotenv

    load_dotenv()

    key_loader = KeyLoader()
    embedder = OpenAIEmbedder(api_key=key_loader.get_openai_api_key())
    # extractor = OpenaiKnowledgeExtractor(openai_api_key=key_loader.get_openai_api_key())
    access_ollama = AccessOllama()
    extractor = OllamaKnowledgeExtractor(access_ollama=access_ollama, model="llama3.1")

    splitter = SplitterChain(splitters=[SectionSplitter(), SemanticSplitter(knowledge_extractor=extractor)],
                             include_all_chunks=True)

    input_document = InputDocument(document_id="doc1", text=input_text, properties={})

    chunks = splitter.split(input_document)

    for chunk in chunks:
        print(f"{chunk.chunk_id} ({chunk.total_chunks}): {chunk.chunk_text}")
