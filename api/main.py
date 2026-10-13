"""Search API.

    uvicorn main:app --app-dir api --port 8000
"""
import time
from contextlib import asynccontextmanager

from fastapi import FastAPI, Query

from search import SearchEngine

engine: SearchEngine | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    engine = SearchEngine()  # load both ONNX models and the DB pool once
    # First ONNX Runtime calls are slow; pay that at startup, not on the first search
    engine.encoder.encode_query("warm up")
    engine.reranker.warm_up()
    yield
    engine.close()


app = FastAPI(title="AI Product Search", lifespan=lifespan)


@app.get("/search")
def search(
    query: str = Query(..., min_length=1, max_length=500),
    top_k: int = Query(10, ge=1, le=50),
):
    start = time.perf_counter()
    results = engine.search(query, top_k=top_k)
    return {
        "query": query,
        "results": results,
        "took_ms": round((time.perf_counter() - start) * 1000),
    }


@app.get("/health")
def health():
    return {"status": "ok"}
