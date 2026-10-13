"""Add full-text search to the products table for the lexical half of hybrid search.

Adds a stored tsvector column (product title weighted above the rest of the
description) and a GIN index on it. Run once after store_embeddings.py; it rewrites
the table (and rebuilds the HNSW index), so it takes a while on a large catalog.

    python indexing/create_text_index.py
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

# The first line of each description is the product title; weight it 'A' so title
# matches outrank matches buried in the bullet points ('B').
ADD_COLUMN_SQL = """
    ALTER TABLE products ADD COLUMN IF NOT EXISTS search_tsv tsvector
    GENERATED ALWAYS AS (
        setweight(to_tsvector('english', split_part(coalesce(description, ''), E'\n', 1)), 'A') ||
        setweight(to_tsvector('english', coalesce(description, '')), 'B')
    ) STORED
"""

CREATE_INDEX_SQL = """
    CREATE INDEX IF NOT EXISTS products_search_tsv_idx ON products USING gin (search_tsv)
"""


def create_text_index():
    connection = psycopg2.connect(**DB_CONFIG)
    connection.autocommit = True
    try:
        with connection.cursor() as cursor:
            # The table rewrite also rebuilds the HNSW index: keep its graph in memory, and
            # avoid parallel builds, which need more than Docker's default 64MB /dev/shm
            cursor.execute("SET maintenance_work_mem = '3GB'")
            cursor.execute("SET max_parallel_maintenance_workers = 0")

            print("Adding search_tsv column (rewrites the table)...")
            start = time.perf_counter()
            cursor.execute(ADD_COLUMN_SQL)
            print(f"Column ready in {time.perf_counter() - start:.0f}s")

            print("Building GIN index...")
            start = time.perf_counter()
            cursor.execute(CREATE_INDEX_SQL)
            print(f"Index ready in {time.perf_counter() - start:.0f}s")

            cursor.execute("ANALYZE products")
    finally:
        connection.close()


if __name__ == "__main__":
    create_text_index()
