"""CPU cross-encoder reranker: ONNX Runtime + tokenizers, no PyTorch.

Uses the ms-marco-MiniLM-L6-v2 model exported by indexing/export_onnx.py. Unlike the
bi-encoder, it reads the query and product text together, so it judges relevance
far more precisely, but is too slow to run over the whole catalog.
"""
from pathlib import Path

import numpy as np
import onnxruntime as ort
from tokenizers import Encoding, Tokenizer

from config import CROSS_ENCODER_DIR, RERANK_BATCH_SIZE, RERANK_MAX_LENGTH


class CrossEncoderReranker:
    def __init__(
        self,
        model_dir: Path = CROSS_ENCODER_DIR,
        max_length: int = RERANK_MAX_LENGTH,
        batch_size: int = RERANK_BATCH_SIZE,
    ):
        self.tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
        # Never truncate the query, only the product text. No padding here: each
        # mini-batch is padded only to its own longest pair in _pad_batch().
        self.tokenizer.enable_truncation(max_length=max_length, strategy="only_second")
        self.tokenizer.no_padding()
        self.batch_size = batch_size

        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        # By default ORT's worker threads keep spinning on the CPU after each call. With
        # two models in one process, each one's spinning threads slow the other down
        # (measured: full search 518 ms with spinning, 233 ms without).
        options.add_session_config_entry("session.intra_op.allow_spinning", "0")
        self.session = ort.InferenceSession(
            str(model_dir / "model.onnx"), options, providers=["CPUExecutionProvider"]
        )

    def warm_up(self):
        """Run one dummy batch: ONNX Runtime's first call is several times slower."""
        self.score("warm up", ["warm up"] * self.batch_size)

    def score(self, query: str, documents: list[str]) -> np.ndarray:
        """Return the raw relevance logit for each document against the query.

        Pairs are sorted by token length and scored in small batches, so short products
        aren't padded up to the length of the longest one. Padding is pure wasted work:
        with one 256-token batch for all candidates, most of the compute went on padding.
        """
        logits = np.empty(len(documents), dtype=np.float32)
        if not documents:
            return logits
        encodings = self.tokenizer.encode_batch([(query, doc) for doc in documents])
        by_length = sorted(range(len(encodings)), key=lambda i: len(encodings[i].ids))
        for start in range(0, len(by_length), self.batch_size):
            batch = by_length[start : start + self.batch_size]
            inputs = _pad_batch([encodings[i] for i in batch])
            logits[batch] = self.session.run(None, inputs)[0]
        return logits

    def rerank(self, query: str, candidates: list[dict], top_k: int, min_score: float = 0.0) -> list[dict]:
        """Sort candidates by cross-encoder relevance, drop weak ones, keep the best top_k.

        Each candidate needs a "description"; a 0-1 "score" key is added to each result.
        """
        logits = self.score(query, [c["description"] for c in candidates])
        # Rank on logits: strong matches all saturate near 1.0 after the sigmoid
        order = np.argsort(-logits)
        relevance = 1.0 / (1.0 + np.exp(-logits))
        ranked = [{**candidates[i], "score": float(relevance[i])} for i in order]
        return [c for c in ranked if c["score"] >= min_score][:top_k]


def _pad_batch(encodings: list[Encoding]) -> dict[str, np.ndarray]:
    """Pad a batch to its longest pair (BERT pads with id 0, mask 0, segment 0)."""
    length = max(len(e.ids) for e in encodings)
    input_ids = np.zeros((len(encodings), length), dtype=np.int64)
    attention_mask = np.zeros_like(input_ids)
    token_type_ids = np.zeros_like(input_ids)
    for row, encoding in enumerate(encodings):
        n = len(encoding.ids)
        input_ids[row, :n] = encoding.ids
        attention_mask[row, :n] = encoding.attention_mask
        token_type_ids[row, :n] = encoding.type_ids
    return {"input_ids": input_ids, "attention_mask": attention_mask, "token_type_ids": token_type_ids}
