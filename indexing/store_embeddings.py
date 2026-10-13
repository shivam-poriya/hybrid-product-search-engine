from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values
import pandas as pd
import torch

# --- Paths ---
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CSV_PATH = DATA_DIR / "products.csv"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.pt"
CHUNK_SIZE = 5000  # rows per CSV chunk

# --- DB config ---
DB_CONFIG = {
    "host": "localhost",
    "database": "product_search",
    "user": "postgres",
    "password": "123",
    "port": "5435",
}


def create_table():
    """Create the products table and pgvector extension if they don't exist."""
    connection = None
    cursor = None
    try:
        connection = psycopg2.connect(**DB_CONFIG)
        cursor = connection.cursor()

        cursor.execute("""
            CREATE EXTENSION IF NOT EXISTS vector;
            CREATE TABLE IF NOT EXISTS products (
                id INT PRIMARY KEY,
                product_id VARCHAR(255) UNIQUE NOT NULL,
                description TEXT,
                embeddings VECTOR(384)
            );
        """)

        connection.commit()
        print("Table created successfully!")

    except Exception as error:
        print(f"Error creating table: {error}")
        if connection:
            connection.rollback()

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()
            print("PostgreSQL connection is closed.")


def store_data():
    """Read products.csv in chunks, pair with embeddings.pt, and batch-insert into PostgreSQL."""
    connection = None
    cursor = None
    try:
        # 1. Load all embeddings into memory
        print("Loading embeddings...")
        embeddings = torch.load(EMBEDDINGS_PATH, map_location="cuda")
        print(f"Embeddings shape: {embeddings.shape}")

        # 2. Connect to PostgreSQL
        connection = psycopg2.connect(**DB_CONFIG)
        cursor = connection.cursor()

        # 3. SQL for batch insert (skip duplicates on conflict)
        insert_sql = """
            INSERT INTO products (id, product_id, description, embeddings)
            VALUES %s
            ON CONFLICT (id) DO NOTHING
        """

        # 4. Read CSV in chunks and insert
        row_offset = 0
        for chunk_num, chunk_df in enumerate(
            pd.read_csv(CSV_PATH, chunksize=CHUNK_SIZE), start=1
        ):
            chunk_size = len(chunk_df)

            # Slice the matching embeddings for this chunk
            chunk_embeddings = embeddings[row_offset : row_offset + chunk_size]

            # Build list of tuples: (id, product_id, description, embedding_vector)
            rows = []
            for i, (_, row) in enumerate(chunk_df.iterrows()):
                row_id = row_offset + i + 1
                product_id = str(row["product_id"])
                description = str(row["combined"]) if pd.notna(row["combined"]) else ""
                # Convert embedding tensor to a pgvector-compatible string "[0.1,0.2,...]"
                emb_list = chunk_embeddings[i].tolist()
                emb_str = "[" + ",".join(f"{v:.8f}" for v in emb_list) + "]"
                rows.append((row_id, product_id, description, emb_str))

            # 5. Batch insert using execute_values (much faster than row-by-row)
            execute_values(cursor, insert_sql, rows, page_size=1000)
            connection.commit()

            row_offset += chunk_size
            print(f"Chunk {chunk_num} done — inserted {row_offset} rows so far")

        print(f"\nAll done! Total rows processed: {row_offset}")

    except Exception as error:
        print(f"Error while storing data: {error}")
        if connection:
            connection.rollback()

    finally:
        if cursor:
            cursor.close()
        if connection:
            connection.close()
            print("PostgreSQL connection is closed.")


if __name__ == "__main__":
    create_table()
    store_data()
