import numpy as np
import onnxruntime as ort

from typing import List
from rag4p.rag.embedding.embedder import Embedder
from tokenizers import Tokenizer


class OnnxEmbedder(Embedder):
    """
    Embedder that uses a ONNX model to embed text. The model is a MiniLM model that has been converted to ONNX. The
    model is loaded from the file all-minilm-l6-v2-q.onnx. The tokenizer is loaded from the file tokenizer.json.
    The model runs on your local machine.
    """
    def __init__(self, max_length: int = 512):
        self.max_length = max_length
        self.tokenizer = Tokenizer.from_file("../data/tokenizer.json")
        self.tokenizer.enable_truncation(max_length=max_length)
        self.tokenizer.enable_padding(pad_to_multiple_of=1)
        self.ort_sess = ort.InferenceSession('../data/all-minilm-l6-v2-q.onnx')

    def embed(self, text: str) -> List[float]:
        tokens = self.tokenizer.encode(text, add_special_tokens=True)

        # Inference
        token_embeddings = self.ort_sess.run(None, {
            'input_ids': (np.array([tokens.ids])),
            'attention_mask': np.array([tokens.attention_mask]),
            'token_type_ids': (np.array([tokens.type_ids])),
        })
        token_embeddings = np.array(token_embeddings)

        # Mean Pooling (note: assumes that there are no padded elements)
        embeddings = np.mean(token_embeddings, axis=2)
        return embeddings[0][0]
