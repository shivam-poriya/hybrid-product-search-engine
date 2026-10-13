"""CPU query encoder: ONNX Runtime + tokenizers, no PyTorch.

Produces the same embeddings as indexing/generate_embeddings.py, using the model
exported by indexing/export_onnx.py.
"""
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Tokenizer

from config import BI_ENCODER_DIR

MAX_LENGTH = 512  # same limit transformers applied when the catalog was embedded


class QueryEncoder:
    def __init__(self, model_dir: Path = BI_ENCODER_DIR):
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=MAX_LENGTH)
        self.tokenizer.enable_padding()

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # By default ORT's worker threads keep spinning on the CPU after each call. With
        # two models in one process, each one's spinning threads slow the other down
        # (measured: full search 518 ms with spinning, 233 ms without).
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.session = ort.InferenceSession(
            str(model_dir / "model.onnx"), options, providers=["CPUExecutionProvider"]
        )

    def encode(self, texts: list[str]) -> np.ndarray:
        """Return a (len(texts), 384) float32 array of sentence embeddings."""
        encodings = self.tokenizer.encode_batch(texts)
        inputs = {
            "input_ids": np.array([e.ids for e in encodings], dtype=np.int64),
            "attention_mask": np.array([e.attention_mask for e in encodings], dtype=np.int64),
            "token_type_ids": np.array([e.type_ids for e in encodings], dtype=np.int64),
        }
        return self.session.run(None, inputs)[0]

    def encode_query(self, query: str) -> list[float]:
        """Embed one query as a plain list, ready to pass to pgvector."""
        return self.encode([query])[0].tolist()
