# RAG4P - Retrieval Augmented Generation for Python
Welcome to the repository for our project [RAG4P.org](https://rag4p.org). This project is a Python implementation of the Retrieval Augmented Generation framework. It is a framework that is simple to use and understand. But powerful 
enough to extend for your own projects.

## Setting up your environment

## Python
We encourage you to use a python environment manager. Poetry makes it easy to use multiple python versions and packages. where you can switch versions per project. Read this [Poetry documentation page](https://python-poetry.org/docs/managing-environments/) to learn how to set up your environment. No poetry installed? Read this page to install it for your environment. [Poetry installation](https://python-poetry.org/docs/#installing-with-the-official-installer)

Setting the right version of python for the project
```bash
poetry env use 3.10
```

Install dependencies
```bash
poetry install
```

Run the project
```bash
poetry run python rag4p/app_step1_chunking_strategy.py
```

## No poetry

Setup your venv
```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies
```bash
pip install -r poetry-requirements.txt
```

## Loading API keys
We try to limit accessing Large Language Models and vector stores to a minimum. You do not need an LLM or vector store to learn about all the elements of the Retrieval Augmented Generation framework, except for the generation part. In the workshop we use the LLM of Open AI, which is not publicly available. We will provide you with a key to access it, if you don't have your own key.

Please use this key for the workshop only, and limit the amount of interaction, or we get blocked for exceeding our
limits. The API key is obtained through a remote file, which is encrypted. Of course you can also use your own key if
you have it.

### Environment variables
The easiest way to load the API key is to set an environment variable for each required key. In Python we prefer the file .env.properties in the root of the project with the following properties:
```properties
openai_api_key=sk-...
weaviate_api_key=...
weaviate_url=...
```

If you do not have your own key, you can load ours. The key is stored in a remote location. You need the .env.properties file in the root of the project with the following line:
```properties
secret_key=...
```
This secret key is used to decrypt the remote file containing the API keys. We will provide the value for this key
during the workshop.
