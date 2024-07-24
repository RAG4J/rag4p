"""
This model provides access to the OpenAI API. It is used to embed text and to generate answers to questions. We also
provide a default model for both embedding and answering.
"""

PROVIDER = "openai"

MODEL_GPT4O = "gpt-4o"
MODEL_GPT4O_MINI = "gpt-4o-mini"
MODEL_GPT4 = "gpt-4"
MODEL_GPT4_TURBO = "gpt-4-turbo"
MODEL_GPT35_TURBO = "gpt-3.5-turbo"

EMBEDDING_ADA = "text-embedding-ada-002"
EMBEDDING_SMALL = "text-embedding-3-small"

DEFAULT_MODEL = MODEL_GPT4O
DEFAULT_EMBEDDING_MODEL = EMBEDDING_SMALL
