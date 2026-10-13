"""Two-stage hybrid search pipeline.

1. Retrieve (hybrid):
   - vector search:  embed the query with the bi-encoder, nearest products from pgvector
   - lexical search: Postgres full-text search, so exact terms (brands, model numbers,
                     sizes) the embedding may blur still get found
   Both lists are merged with Reciprocal Rank Fusion into one pre-ranked list.
2. Rerank: score the top fused candidates with the cross-encoder and keep the best.
"""
import time
from concurrent.futures import ThreadPoolExecutor

import psycopg2
import psycopg2.errors
from psycopg2.pool import ThreadedConnectionPool

from config import (
    CANDIDATE_POOL,
    DB_CONFIG,
    HNSW_EF_SEARCH,
    LEXICAL_POOL,
    LEXICAL_TIMEOUT_MS,
    MIN_RELEVANCE,
    RRF_K,
    VECTOR_POOL,
)
from encoder import QueryEncoder
from reranker import CrossEncoderReranker

VECTOR_SEARCH_SQL = """
    SELECT product_id, description, 1 - (embeddings <=> %(vector)s::vector) AS similarity
    FROM products
    ORDER BY embeddings <=> %(vector)s::vector
    LIMIT %(limit)s
"""

# Products containing every query term (title words weighted A rank higher).
# Terms no product contains (typos like "ploroid") are dropped first; otherwise one typo
# leaves zero matches and forces the slow any-term fallback. Each existence check is a
# single GIN index lookup. Normalization 1 stops long descriptions winning on length.
LEXICAL_ALL_TERMS_SQL = """
    WITH known AS (
        SELECT lexeme
        FROM unnest(tsvector_to_array(to_tsvector('english', %(query)s))) AS lexeme
        WHERE EXISTS (
            SELECT 1 FROM products WHERE search_tsv @@ to_tsquery('simple', quote_literal(lexeme))
        )
    ),
    q AS (
        SELECT to_tsquery('simple', string_agg(quote_literal(lexeme), ' & ')) AS tsq FROM known
    )
    SELECT product_id, description, ts_rank(search_tsv, q.tsq, 1) AS rank
    FROM products, q
    WHERE search_tsv @@ q.tsq
    ORDER BY rank DESC
    LIMIT %(limit)s
"""

# Fallback: products containing any query term. Ranking every match of a common word is
# slow (0.4 s warm, several seconds on a cold cache), so it only runs when the all-terms
# query finds too few products, and only within the keyword search time budget.
# The lexemes are already stemmed, so re-parse them with 'simple' to avoid stemming twice.
LEXICAL_ANY_TERM_SQL = """
    WITH q AS (
        SELECT to_tsquery('simple', replace(plainto_tsquery('english', %(query)s)::text, ' & ', ' | ')) AS tsq
    )
    SELECT product_id, description, ts_rank(search_tsv, q.tsq, 1) AS rank
    FROM products, q
    WHERE search_tsv @@ q.tsq
    ORDER BY rank DESC
    LIMIT %(limit)s
"""


class SearchEngine:
    def __init__(self):
        self.encoder = QueryEncoder()
        self.reranker = CrossEncoderReranker()
        self.pool = ThreadedConnectionPool(1, 20, **DB_CONFIG)
        # Runs the vector and lexical queries concurrently
        self.executor = ThreadPoolExecutor(max_workers=8)

    def close(self):
        self.executor.shutdown()
        self.pool.closeall()

    def _fetch(self, statements: list[tuple[str, dict]], retry: bool = True) -> list[tuple]:
        """Run statements in one transaction on a pooled connection; return the last result."""
        connection = self.pool.getconn()
        broken = False
        try:
            with connection.cursor() as cursor:
                for sql, params in statements:
                    cursor.execute(sql, params)
                rows = cursor.fetchall()
            connection.commit()
        except psycopg2.errors.QueryCanceled:
            # statement_timeout fired; the connection is fine. (Must come before the
            # OperationalError branch, which QueryCanceled is a subclass of.)
            connection.rollback()
            raise
        except (psycopg2.OperationalError, psycopg2.InterfaceError):
            # Connection died (e.g. the database restarted): discard it and retry once
            # on a fresh one, instead of failing this request
            broken = True
        except psycopg2.Error:
            connection.rollback()
            raise
        finally:
            self.pool.putconn(connection, close=broken or connection.closed)
        if broken:
            if not retry:
                raise psycopg2.OperationalError("Database connection lost")
            return self._fetch(statements, retry=False)
        return rows

    def vector_search(self, query: str, limit: int) -> list[dict]:
        vector = "[" + ",".join(f"{v:.8f}" for v in self.encoder.encode_query(query)) + "]"
        rows = self._fetch([
            # The HNSW index returns at most ef_search rows, so widen it for big pools
            ("SET LOCAL hnsw.ef_search = %(ef)s", {"ef": max(HNSW_EF_SEARCH, limit)}),
            (VECTOR_SEARCH_SQL, {"vector": vector, "limit": limit}),
        ])
        return [
            {"product_id": product_id, "description": description or "", "vector_score": float(similarity)}
            for product_id, description, similarity in rows
        ]

    def _fetch_within(self, sql: str, params: dict, deadline: float) -> list[tuple]:
        """Run a keyword query, giving up (returning nothing) once the deadline passes.

        Keyword search only adds candidates to vector search, so a slow keyword query
        should cost some recall, never seconds of latency.
        """
        remaining_ms = int((deadline - time.perf_counter()) * 1000)
        if remaining_ms < 20:
            return []
        try:
            return self._fetch([
                ("SET LOCAL statement_timeout = %(ms)s", {"ms": remaining_ms}),
                (sql, params),
            ])
        except psycopg2.errors.QueryCanceled:
            return []

    def lexical_search(self, query: str, limit: int) -> list[dict]:
        params = {"query": query, "limit": limit}
        deadline = time.perf_counter() + LEXICAL_TIMEOUT_MS / 1000
        rows = self._fetch_within(LEXICAL_ALL_TERMS_SQL, params, deadline)
        if len(rows) < limit:
            # Too few products contain every term: top up with partial matches
            found = {row[0] for row in rows}
            rows += [
                row for row in self._fetch_within(LEXICAL_ANY_TERM_SQL, params, deadline)
                if row[0] not in found
            ]
        return [
            {"product_id": product_id, "description": description or "", "lexical_score": float(rank)}
            for product_id, description, rank in rows[:limit]
        ]

    def hybrid_search(self, query: str, vector_pool: int, lexical_pool: int) -> list[dict]:
        """Run both retrievers in parallel and fuse them into one pre-ranked list."""
        vector_future = self.executor.submit(self.vector_search, query, vector_pool)
        lexical_future = self.executor.submit(self.lexical_search, query, lexical_pool)
        return reciprocal_rank_fusion(vector_future.result(), lexical_future.result())

    def search(
        self,
        query: str,
        top_k: int = 10,
        candidate_pool: int = CANDIDATE_POOL,
        min_relevance: float = MIN_RELEVANCE,
    ) -> list[dict]:
        # Always give the reranker more candidates than it has to return
        candidate_pool = max(candidate_pool, 2 * top_k)
        candidates = self.hybrid_search(
            query, max(VECTOR_POOL, candidate_pool), max(LEXICAL_POOL, candidate_pool)
        )
        # Pre-ranked by fusion, so only the most promising reach the (slow) cross-encoder
        candidates = deduplicate(candidates)[:candidate_pool]
        return self.reranker.rerank(query, candidates, top_k, min_relevance)


def reciprocal_rank_fusion(*ranked_lists: list[dict], k: int = RRF_K) -> list[dict]:
    """Merge ranked lists: each product scores sum(1 / (k + rank)) over the lists it is in.

    Uses only rank positions, so cosine similarities and ts_rank values never need to be
    put on a common scale, and products found by both retrievers rise to the top.
    """
    fused: dict[str, dict] = {}
    for ranked in ranked_lists:
        for rank, item in enumerate(ranked, start=1):
            entry = fused.setdefault(item["product_id"], {"fusion_score": 0.0})
            entry.update(item)
            entry["fusion_score"] += 1.0 / (k + rank)
    return sorted(fused.values(), key=lambda c: c["fusion_score"], reverse=True)


def deduplicate(candidates: list[dict]) -> list[dict]:
    """Drop listings whose description repeats an earlier (better-ranked) one; the catalog has
    many identical products under different IDs, and reranking them is wasted work."""
    seen = set()
    unique = []
    for candidate in candidates:
        key = candidate["description"].strip().lower()
        if key not in seen:
            seen.add(key)
            unique.append(candidate)
    return unique
