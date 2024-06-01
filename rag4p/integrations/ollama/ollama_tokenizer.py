from rag4p.integrations.ollama import EMBEDDING_MODEL_NOMIC, EMBEDDING_MODEL_NOMIC_TOKENIZER, EMBEDDING_MODEL_MINILM, \
    EMBEDDING_MODEL_MINILM_TOKENIZER


def tokenizer_for_model(embedder_model_name: str):
    """
    Returns the tokenizer for the given embedder model name. First check the available models in the ollama module.
    Then look for the Hugging Face models for that specific model. Return the tokenizer to be used.
    :param embedder_model_name:
    :return:
    """
    if embedder_model_name == EMBEDDING_MODEL_NOMIC:
        return EMBEDDING_MODEL_NOMIC_TOKENIZER
    elif embedder_model_name == EMBEDDING_MODEL_MINILM:
        return EMBEDDING_MODEL_MINILM_TOKENIZER
    else:
        raise ValueError(f"Unknown embedder model name: {embedder_model_name}")