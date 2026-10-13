"""Settings for the search API. Every value can be overridden with an environment variable."""
import os
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
BI_ENCODER_DIR = MODELS_DIR / "all-MiniLM-L6-v2-onnx"
CROSS_ENCODER_DIR = MODELS_DIR / "ms-marco-MiniLM-L6-v2-onnx"

DB_CONFIG = {
    "host": os.getenv("DB_HOST", "localhost"),
    "port": os.getenv("DB_PORT", "5435"),
    "database": os.getenv("DB_NAME", "product_search"),
    "user": os.getenv("DB_USER", "postgres"),
    "password": os.getenv("DB_PASSWORD", "123"),
}

# Hits pulled from each retriever in the hybrid first stage
VECTOR_POOL = int(os.getenv("VECTOR_POOL", "50"))
LEXICAL_POOL = int(os.getenv("LEXICAL_POOL", "50"))
# Time budget for keyword search; past it, whatever was found so far is used. Common
# words on a cold cache can otherwise take seconds.
LEXICAL_TIMEOUT_MS = int(os.getenv("LEXICAL_TIMEOUT_MS", "300"))
# Reciprocal Rank Fusion constant: higher flattens the gap between top and lower ranks
RRF_K = int(os.getenv("RRF_K", "60"))
# How many fused candidates the cross-encoder gets to rerank
CANDIDATE_POOL = int(os.getenv("CANDIDATE_POOL", "50"))
# Tokens of (query + product text) the cross-encoder reads. Product descriptions are
# long; 256 keeps title + key bullet points. 128 is ~2x faster with slightly worse ranking.
RERANK_MAX_LENGTH = int(os.getenv("RERANK_MAX_LENGTH", "128"))
# Candidates are sorted by length and scored in batches of this size, each padded only
# to its own longest pair instead of all 50 padded to the longest one
RERANK_BATCH_SIZE = int(os.getenv("RERANK_BATCH_SIZE", "8"))
# HNSW search breadth; must be >= VECTOR_POOL or the index returns fewer rows
HNSW_EF_SEARCH = int(os.getenv("HNSW_EF_SEARCH", "100"))
# Cross-encoder relevance (0-1) below which a candidate is dropped
MIN_RELEVANCE = float(os.getenv("MIN_RELEVANCE", "0.0"))
