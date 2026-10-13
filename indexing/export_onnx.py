"""Export the search models to ONNX so queries can be served on CPU without PyTorch.

- Bi-encoder (all-MiniLM-L6-v2): mean pooling is baked into the graph, so it returns
  sentence embeddings identical to the ones in generate_embeddings.py.
- Cross-encoder (ms-marco-MiniLM-L6-v2): scores (query, product) pairs to rerank
  the vector search candidates. Exported with eager attention and then fused with
  ONNX Runtime's BERT optimizer, which makes it ~2.5x faster on CPU with identical scores.

Run once (needs torch + transformers + onnx + onnxscript):

    python indexing/export_onnx.py
"""
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from onnxruntime.transformers.optimizer import optimize_model
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

BI_ENCODER_CKPT = "sentence-transformers/all-MiniLM-L6-v2"
BI_ENCODER_DIR = MODELS_DIR / "all-MiniLM-L6-v2-onnx"

CROSS_ENCODER_CKPT = "cross-encoder/ms-marco-MiniLM-L6-v2"
CROSS_ENCODER_DIR = MODELS_DIR / "ms-marco-MiniLM-L6-v2-onnx"

INPUT_NAMES = ["input_ids", "attention_mask", "token_type_ids"]
DYNAMIC_INPUT_AXES = {name: {0: "batch", 1: "sequence"} for name in INPUT_NAMES}


class MeanPooledEncoder(torch.nn.Module):
    """Transformer + the same mean pooling used in generate_embeddings.py."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, token_type_ids):
        token_embeddings = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * mask, dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)


class CrossEncoderScorer(torch.nn.Module):
    """Sequence classifier returning one relevance logit per (query, product) pair."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, input_ids, attention_mask, token_type_ids):
        logits = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        ).logits
        return logits.squeeze(-1)


def fuse_bert_graph(onnx_path, config):
    """Rewrite the graph in place with ONNX Runtime's fused transformer ops.

    Each layer's ~20 small ops (MatMul/Softmax/Add/LayerNorm...) are replaced by a few
    fused kernels (Attention, SkipLayerNormalization, BiasGelu, EmbedLayerNormalization)
    that do the same math with far less overhead and memory traffic.
    """
    optimized = optimize_model(
        str(onnx_path),
        model_type="bert",
        num_heads=config.num_attention_heads,
        hidden_size=config.hidden_size,
        opt_level=0,  # only the Python-side fusions; ORT applies its own at load time
    )
    fused = {op: count for op, count in optimized.get_fused_operator_statistics().items() if count}
    print(f"Fused operators: {fused}")
    optimized.save_model_to_file(str(onnx_path))


def export_model(module, tokenizer, output_dir, output_name, sample, verify_inputs, fuse=False):
    output_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = output_dir / "model.onnx"
    module.eval()

    torch.onnx.export(
        module,
        tuple(sample[name] for name in INPUT_NAMES),
        str(onnx_path),
        input_names=INPUT_NAMES,
        output_names=[output_name],
        dynamic_axes={**DYNAMIC_INPUT_AXES, output_name: {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    if fuse:
        fuse_bert_graph(onnx_path, module.model.config)
    # tokenizer.json is all the lightweight `tokenizers` library needs at query time
    tokenizer.save_pretrained(output_dir)
    print(f"Saved ONNX model and tokenizer to {output_dir}")

    # Check the (possibly fused) ONNX output matches PyTorch on inputs of a different shape
    with torch.no_grad():
        expected = module(*(verify_inputs[name] for name in INPUT_NAMES)).numpy()
    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    actual = session.run(None, {name: verify_inputs[name].numpy() for name in INPUT_NAMES})[0]

    max_diff = np.abs(expected - actual).max()
    print(f"Max difference vs PyTorch: {max_diff:.2e}")
    assert max_diff < 1e-4, "ONNX output does not match PyTorch"


def export_bi_encoder():
    tokenizer = AutoTokenizer.from_pretrained(BI_ENCODER_CKPT)
    model = MeanPooledEncoder(AutoModel.from_pretrained(BI_ENCODER_CKPT))

    sample = tokenizer(["export sample", "a slightly longer export sample"], padding=True, return_tensors="pt")
    verify = tokenizer(
        ["quiet bathroom exhaust fan", "wireless noise cancelling headphones for travel", "gift"],
        padding=True, truncation=True, return_tensors="pt",
    )
    export_model(model, tokenizer, BI_ENCODER_DIR, "sentence_embedding", sample, verify)


def export_cross_encoder():
    tokenizer = AutoTokenizer.from_pretrained(CROSS_ENCODER_CKPT)
    # Eager attention exports as plain MatMul/Softmax ops that the BERT optimizer can fuse
    # into one Attention op. The default (SDPA) exports a graph that can't be fused and
    # runs ~2x slower in ONNX Runtime.
    model = CrossEncoderScorer(
        AutoModelForSequenceClassification.from_pretrained(CROSS_ENCODER_CKPT, attn_implementation="eager")
    )

    sample = tokenizer(
        ["export sample", "export sample"],
        ["a product description", "a slightly longer product description for export"],
        padding=True, return_tensors="pt",
    )
    verify = tokenizer(
        ["quiet bathroom exhaust fan"] * 3,
        [
            "Delta BreezSignature 80 CFM Exhaust Bath Fan with Humidity Sensor, virtually silent",
            "Stainless steel kitchen knife set with wooden block",
            "Ceiling fan",
        ],
        padding=True, truncation=True, return_tensors="pt",
    )
    export_model(model, tokenizer, CROSS_ENCODER_DIR, "relevance_logit", sample, verify, fuse=True)


if __name__ == "__main__":
    export_bi_encoder()
    export_cross_encoder()
