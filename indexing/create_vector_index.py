"""Build an HNSW index on the product embeddings so vector search doesn't scan every row.

Without it, each query compares against all ~1.2M products (seconds per search).
Run once after store_embeddings.py; the build takes a while on a large catalog.

    python indexing/create_vector_index.py
"""
import time

import psycopg2

DB_CONFIG = {
    "host": "localhost",
    "database": "product_search",
    "user": "postgres",
    "password": "123",
    "port": "5435",
}

# Cosine distance, matching the `<=>` operator used by the search API
CREATE_INDEX_SQL = """
    CREATE INDEX IF NOT EXISTS products_embeddings_hnsw_idx
    ON products USING hnsw (embeddings vector_cosine_ops)
    WITH (m = 16, ef_construction = 64)
"""


def create_index():
    connection = psycopg2.connect(**DB_CONFIG)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            # Keep the graph in memory while building; much faster than spilling to disk
            cursor.execute("SET maintenance_work_mem = '3GB'")
            # Parallel builds need shared memory beyond Docker's default 64MB /dev/shm
            cursor.execute("SET max_parallel_maintenance_workers = 0")
            print("Building HNSW index...")
            start = time.perf_counter()
            cursor.execute(CREATE_INDEX_SQL)
            print(f"Index ready in {time.perf_counter() - start:.0f}s")
    finally:
        connection.close()


if __name__ == "__main__":
    create_index()
