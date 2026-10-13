# AI-Based Search System for E-Commerce

Product search with hybrid retrieval and cross-encoder reranking:

1. **Retrieve (hybrid)** – two searches run in parallel:
   - *vector*: the query is embedded with `sentence-transformers/all-MiniLM-L6-v2` and the
     nearest products are pulled from PostgreSQL + pgvector (HNSW index);
   - *keyword*: Postgres full-text search (GIN index, title words weighted higher) catches
     exact terms such as brands and model numbers.
2. **Fuse** – Reciprocal Rank Fusion merges both lists into one pre-ranked, de-duplicated list.
3. **Rerank** – `cross-encoder/ms-marco-MiniLM-L6-v2` scores the top fused candidates against
   the query and shortlists the best results.

Both models run on CPU through ONNX Runtime; PyTorch is only needed offline. The cross-encoder is
exported with eager attention and fused with ONNX Runtime's BERT optimizer, so a full search takes ~0.25 s.

## Project structure

```
searchSystem/
├── data/                     # products.csv, embeddings.pt (not committed)
├── indexing/                 # Offline pipeline that builds the search index
│   ├── generate_embeddings.py    # products.csv -> embeddings.pt
│   ├── store_embeddings.py       # embeddings.pt -> Postgres (pgvector)
│   ├── create_vector_index.py    # HNSW index for fast vector search
│   ├── create_text_index.py      # Full-text column + GIN index for keyword search
│   └── export_onnx.py            # both models -> models/ (ONNX) for CPU serving
├── models/                   # Exported ONNX models + tokenizers (not committed)
├── api/                      # Search API (no PyTorch)
│   ├── main.py                   # FastAPI app: GET /search
│   ├── search.py                 # Hybrid retrieve + fuse + rerank pipeline
│   ├── encoder.py                # Bi-encoder: query -> embedding
│   ├── reranker.py               # Cross-encoder: (query, product) -> relevance
│   └── config.py                 # Settings (env-overridable)
├── frontend/                 # Streamlit search UI
│   ├── .streamlit/config.toml
│   └── app.py
├── docker-compose.yml        # Postgres + pgvector (port 5435)
├── requirements.txt          # All Python dependencies
└── README.md
```

## Getting started

```bash
pip install -r requirements.txt

# 1. Database
docker compose up -d

# 2. Build the index (one-time)
python indexing/generate_embeddings.py
python indexing/store_embeddings.py
python indexing/create_vector_index.py
python indexing/create_text_index.py
python indexing/export_onnx.py

# 3. Search API
uvicorn main:app --app-dir api --port 8002

# 4. Frontend (in another terminal)
streamlit run frontend/app.py
```

## Configuration

Set as environment variables before starting the API:

| Variable | Default | Meaning |
|---|---|---|
| `VECTOR_POOL` | 50 | Hits taken from vector search |
| `LEXICAL_POOL` | 50 | Hits taken from keyword search |
| `LEXICAL_TIMEOUT_MS` | 300 | Time budget for keyword search; past it, results found so far are used |
| `RRF_K` | 60 | Reciprocal Rank Fusion constant |
| `CANDIDATE_POOL` | 50 | Top fused candidates passed to the cross-encoder |
| `RERANK_MAX_LENGTH` | 128 | Tokens the cross-encoder reads per pair (product text is truncated, the query never is) |
| `RERANK_BATCH_SIZE` | 8 | Candidates are sorted by length and scored in batches of this size, each padded only to its own longest |
| `MIN_RELEVANCE` | 0.0 | Drop results whose cross-encoder relevance (0–1) is below this |
| `HNSW_EF_SEARCH` | 100 | HNSW search breadth (higher = better recall, slower) |
| `DB_HOST`, `DB_PORT`, `DB_NAME`, `DB_USER`, `DB_PASSWORD` | see `api/config.py` | Database connection |

The frontend calls the API at `SEARCH_API_URL` (default `http://127.0.0.1:8002/search`). Use `127.0.0.1`,
not `localhost`: on Windows `localhost` adds a ~2 s delay to every request (IPv6 is tried first).
