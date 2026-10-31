from __future__ import annotations

import csv
import argparse
import hashlib
import os
import tempfile
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / "dataset" / "interactions.csv"
COHORT_PATH = BASE_DIR / "cohort_preferences.json"


def get_database_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.environ["DB_DATABASE"],
        user=os.environ["DB_USERNAME"],
        password=os.environ["DB_PASSWORD"],
        connect_timeout=5,
    )


def get_search_index_connection():
    return psycopg2.connect(
        host=os.getenv("ML_INDEX_DB_HOST") or os.getenv("DB_HOST", "localhost"),
        port=os.getenv("ML_INDEX_DB_PORT") or os.getenv("DB_PORT", "5432"),
        dbname=os.getenv("ML_INDEX_DB_DATABASE") or os.environ["DB_DATABASE"],
        user=os.getenv("ML_INDEX_DB_USERNAME") or os.environ["DB_USERNAME"],
        password=os.getenv("ML_INDEX_DB_PASSWORD") or os.environ["DB_PASSWORD"],
        connect_timeout=5,
    )


def snapshot_user_histories(allow_empty: bool = False) -> int:
    load_dotenv(BASE_DIR / ".env")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with get_database_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT accounts.ml_user_token, histories.isbn, histories.rating
                FROM user_histories histories
                JOIN accounts ON accounts.id = histories.user_id
                WHERE accounts.ml_user_token IS NOT NULL
                  AND btrim(accounts.ml_user_token) <> ''
                ORDER BY histories.id
                """
            )
            rows = cursor.fetchall()

    if not rows and not allow_empty:
        raise ValueError("No user histories found; existing interaction data was left unchanged.")
    interactions = {
        (str(user_key), str(isbn)): int(rating)
        for user_key, isbn, rating in rows
    }

    with tempfile.NamedTemporaryFile(
        mode="w", newline="", encoding="utf-8", dir=OUTPUT_PATH.parent, delete=False
    ) as temporary_file:
        writer = csv.writer(temporary_file)
        writer.writerow(["User-Key", "ISBN", "rating"])
        writer.writerows((*key, rating) for key, rating in interactions.items())
        temporary_path = Path(temporary_file.name)

    temporary_path.replace(OUTPUT_PATH)
    catalog_count = snapshot_book_catalog()
    print(f"Indexed {catalog_count} catalog records in PostgreSQL")
    cohort_count = snapshot_cohort_preferences()
    print(f"Saved {cohort_count} cohort preferences to {COHORT_PATH}")
    return len(interactions)


def snapshot_book_catalog() -> int:
    """Stream catalog rows into a PostgreSQL vector index in bounded batches."""
    load_dotenv(BASE_DIR / ".env")
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")
    indexed_count = 0
    with get_search_index_connection() as index_connection:
        with index_connection.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS ml_book_embeddings (
                    book_detail_id BIGINT PRIMARY KEY,
                    isbn TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content_hash CHAR(64) NOT NULL,
                    embedding vector(384) NOT NULL
                )
                """
            )
            cursor.execute(
                "CREATE TEMP TABLE seen_ml_books (book_detail_id BIGINT PRIMARY KEY) "
                "ON COMMIT PRESERVE ROWS"
            )
        index_connection.commit()

        with get_database_connection() as source_connection:
            with source_connection.cursor(name="catalog_snapshot") as source:
                source.execute(
                    """
                    WITH category_agg AS (
                        SELECT bdc.book_data_id, string_agg(DISTINCT c.name, ' ') AS categories
                        FROM book_data_category bdc
                        JOIN categories c ON c.id = bdc.category_id AND c.deleted_at IS NULL
                        GROUP BY bdc.book_data_id
                    ), subject_agg AS (
                        SELECT bds.book_data_id, string_agg(DISTINCT s.name, ' ') AS subjects
                        FROM book_data_subject bds
                        JOIN subjects s ON s.id = bds.subject_id AND s.deleted_at IS NULL
                        GROUP BY bds.book_data_id
                    ), author_agg AS (
                        SELECT bda.book_data_id,
                               string_agg(DISTINCT concat_ws(' ', a.first_name, a.last_name), ' ') AS authors
                        FROM book_data_author bda
                        JOIN authors a ON a.id = bda.author_id AND a.deleted_at IS NULL
                        GROUP BY bda.book_data_id
                    )
                    SELECT bd.id, bd.isbn, data.book_title, data.subtitle, data.description,
                           data.note, bd.classification, bd.call_number,
                           COALESCE(category_agg.categories, '') AS categories,
                           COALESCE(subject_agg.subjects, '') AS subjects,
                           COALESCE(author_agg.authors, '') AS authors
                    FROM book_details bd
                    JOIN book_datas data ON data.id = bd.book_data_id
                    LEFT JOIN category_agg ON category_agg.book_data_id = data.id
                    LEFT JOIN subject_agg ON subject_agg.book_data_id = data.id
                    LEFT JOIN author_agg ON author_agg.book_data_id = data.id
                    WHERE bd.isbn IS NOT NULL AND btrim(bd.isbn) <> ''
                      AND bd.deleted_at IS NULL AND data.deleted_at IS NULL
                    ORDER BY bd.id
                    """
                )
                while rows := source.fetchmany(128):
                    ids = [row[0] for row in rows]
                    with index_connection.cursor() as index_cursor:
                        execute_values(
                            index_cursor,
                            "INSERT INTO seen_ml_books (book_detail_id) VALUES %s ON CONFLICT DO NOTHING",
                            [(book_id,) for book_id in ids],
                        )
                        index_cursor.execute(
                            "SELECT book_detail_id, content_hash FROM ml_book_embeddings WHERE book_detail_id = ANY(%s)",
                            (ids,),
                        )
                        old_hashes = dict(index_cursor.fetchall())
                        changed = []
                        documents = []
                        for row in rows:
                            book_id, isbn, title, *metadata = row
                            document = " ".join(str(value or "") for value in (title, *metadata))
                            digest = hashlib.sha256(
                                (str(isbn) + "\0" + document).encode("utf-8")
                            ).hexdigest()
                            if old_hashes.get(book_id) != digest:
                                changed.append((book_id, isbn, title or "Untitled", digest))
                                documents.append(document)

                        if changed:
                            vectors = model.encode(
                                documents, normalize_embeddings=True, show_progress_bar=False
                            )
                            records = [
                                (*book, "[" + ",".join(f"{value:.8f}" for value in vector) + "]")
                                for book, vector in zip(changed, vectors)
                            ]
                            execute_values(
                                index_cursor,
                                """INSERT INTO ml_book_embeddings
                                   (book_detail_id, isbn, title, content_hash, embedding) VALUES %s
                                   ON CONFLICT (book_detail_id) DO UPDATE SET isbn = EXCLUDED.isbn,
                                   title = EXCLUDED.title, content_hash = EXCLUDED.content_hash,
                                   embedding = EXCLUDED.embedding""",
                                records,
                                template="(%s, %s, %s, %s, %s::vector)",
                            )
                    index_connection.commit()
                    indexed_count += len(rows)

        if indexed_count == 0:
            raise ValueError("Catalog snapshot returned no books; existing search index was left unchanged.")

        with index_connection.cursor() as cursor:
            cursor.execute(
                "DELETE FROM ml_book_embeddings WHERE NOT EXISTS "
                "(SELECT 1 FROM seen_ml_books seen WHERE seen.book_detail_id = ml_book_embeddings.book_detail_id)"
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS ml_book_embeddings_hnsw "
                "ON ml_book_embeddings USING hnsw (embedding vector_cosine_ops)"
            )
        index_connection.commit()
    return indexed_count


def snapshot_cohort_preferences() -> int:
    """Export only program/year preferences shared by at least three students."""
    with get_database_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT lower(btrim(students.program)), lower(btrim(students.year_level)),
                       btrim(histories.isbn), avg(histories.rating)::float,
                       count(*)::int, count(DISTINCT histories.user_id)::int
                FROM user_histories histories
                JOIN students ON students.account_id = histories.user_id AND students.deleted_at IS NULL
                WHERE students.program IS NOT NULL AND btrim(students.program) <> ''
                  AND students.year_level IS NOT NULL AND btrim(students.year_level) <> ''
                  AND histories.isbn IS NOT NULL AND btrim(histories.isbn) <> ''
                GROUP BY lower(btrim(students.program)), lower(btrim(students.year_level)),
                         btrim(histories.isbn)
                HAVING count(DISTINCT histories.user_id) >= 3
                """
            )
            rows = [
                {
                    "program": row[0], "year_level": row[1], "isbn": row[2],
                    "mean_rating": row[3], "interactions": row[4], "students": row[5],
                }
                for row in cursor.fetchall()
            ]

    temporary_path = COHORT_PATH.with_suffix(".tmp")
    temporary_path.write_text(json.dumps(rows), encoding="utf-8")
    temporary_path.replace(COHORT_PATH)
    return len(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Write an empty CSV when user_histories has no rows.",
    )
    count = snapshot_user_histories(allow_empty=parser.parse_args().allow_empty)
    print(f"Added {count} new user histories to {OUTPUT_PATH}")
