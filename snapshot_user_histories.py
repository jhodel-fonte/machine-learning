from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path

import psycopg2
from psycopg2.extras import execute_values
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
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
    with get_database_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                "SELECT count(*) FROM user_histories histories JOIN accounts ON accounts.id = histories.user_id "
                "WHERE accounts.ml_user_token IS NOT NULL AND btrim(accounts.ml_user_token) <> ''"
            )
            interaction_count = cursor.fetchone()[0]
    if not interaction_count and not allow_empty:
        raise ValueError("No user histories found in the database.")
    catalog_count = snapshot_book_catalog()
    print(f"Indexed {catalog_count} catalog records in PostgreSQL")
    stats_count = snapshot_year_relevance()
    print(f"Saved {stats_count} year/program book scores to the ML database")
    return interaction_count


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
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS ml_book_embeddings_isbn_norm_idx "
                "ON ml_book_embeddings (upper(regexp_replace(btrim(isbn), '[-[:space:]]', '', 'g')))"
            )
        index_connection.commit()
    return indexed_count


def snapshot_year_relevance() -> int:
    """Store smoothed borrower relevance from loan transactions with issue-time cohorts."""
    with get_database_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """WITH loan_events AS (
                       SELECT DISTINCT tx.school_id AS student_id,
                              upper(regexp_replace(btrim(bd.isbn), '[-[:space:]]', '', 'g')) AS isbn
                       FROM borrowing_transactions tx
                       JOIN books b ON b.id = tx.book_id
                       JOIN book_details bd ON bd.id = b.book_detail_id
                       JOIN borrow_types bt ON bt.id = tx.borrow_type_id
                       WHERE tx.issued_date >= CURRENT_DATE - INTERVAL '365 days'
                         AND lower(bt.name) <> 'room use only'
                         AND bd.isbn IS NOT NULL AND btrim(bd.isbn) <> ''
                   )
                   SELECT isbn, count(DISTINCT student_id)::int,
                          (SELECT count(DISTINCT student_id) FROM loan_events)::int
                   FROM loan_events GROUP BY isbn"""
            )
            popularity_counts = cursor.fetchall()
            cursor.execute(
                """
                WITH loan_events AS (
                    SELECT DISTINCT lower(btrim(tx.program_at_issue)) AS program,
                           lower(btrim(tx.year_level_at_issue)) AS year_level,
                           tx.school_id AS student_id,
                           upper(regexp_replace(btrim(bd.isbn), '[-[:space:]]', '', 'g')) AS isbn
                    FROM borrowing_transactions tx
                    JOIN books b ON b.id = tx.book_id
                    JOIN book_details bd ON bd.id = b.book_detail_id
                    JOIN borrow_types bt ON bt.id = tx.borrow_type_id
                    WHERE tx.issued_date >= CURRENT_DATE - INTERVAL '365 days'
                      AND lower(bt.name) <> 'room use only'
                      AND tx.program_at_issue IS NOT NULL AND btrim(tx.program_at_issue) <> ''
                      AND tx.year_level_at_issue IS NOT NULL AND btrim(tx.year_level_at_issue) <> ''
                      AND bd.isbn IS NOT NULL AND btrim(bd.isbn) <> ''
                ), group_sizes AS (
                    SELECT program, year_level, count(DISTINCT student_id)::int AS active_borrowers
                    FROM loan_events GROUP BY program, year_level
                ), group_books AS (
                    SELECT program, year_level, isbn, count(DISTINCT student_id)::int AS borrowers
                    FROM loan_events GROUP BY program, year_level, isbn
                ), book_totals AS (
                    SELECT isbn, sum(borrowers)::int AS borrowers FROM group_books GROUP BY isbn
                ), total_students AS (
                    SELECT sum(active_borrowers)::int AS active_borrowers FROM group_sizes
                )
                SELECT gb.program, gb.year_level, gb.isbn, gb.borrowers, gs.active_borrowers,
                       (bt.borrowers - gb.borrowers)::int AS other_borrowers,
                       (ts.active_borrowers - gs.active_borrowers)::int AS other_active_borrowers
                FROM group_books gb
                JOIN group_sizes gs USING (program, year_level)
                JOIN book_totals bt USING (isbn)
                CROSS JOIN total_students ts
                WHERE gb.borrowers >= 3
                """
            )
            counts = cursor.fetchall()
            cursor.execute(
                """
                WITH loan_events AS (
                    SELECT DISTINCT lower(btrim(tx.year_level_at_issue)) AS year_level,
                           tx.school_id AS student_id,
                           upper(regexp_replace(btrim(bd.isbn), '[-[:space:]]', '', 'g')) AS isbn
                    FROM borrowing_transactions tx
                    JOIN books b ON b.id = tx.book_id
                    JOIN book_details bd ON bd.id = b.book_detail_id
                    JOIN borrow_types bt ON bt.id = tx.borrow_type_id
                    WHERE tx.issued_date >= CURRENT_DATE - INTERVAL '365 days'
                      AND lower(bt.name) <> 'room use only'
                      AND tx.year_level_at_issue IS NOT NULL AND btrim(tx.year_level_at_issue) <> ''
                      AND bd.isbn IS NOT NULL AND btrim(bd.isbn) <> ''
                ), group_sizes AS (
                    SELECT year_level, count(DISTINCT student_id)::int AS active_borrowers
                    FROM loan_events GROUP BY year_level
                ), group_books AS (
                    SELECT year_level, isbn, count(DISTINCT student_id)::int AS borrowers
                    FROM loan_events GROUP BY year_level, isbn
                ), book_totals AS (
                    SELECT isbn, sum(borrowers)::int AS borrowers FROM group_books GROUP BY isbn
                ), total_students AS (
                    SELECT sum(active_borrowers)::int AS active_borrowers FROM group_sizes
                )
                SELECT '*' AS program, gb.year_level, gb.isbn, gb.borrowers, gs.active_borrowers,
                       (bt.borrowers - gb.borrowers)::int AS other_borrowers,
                       (ts.active_borrowers - gs.active_borrowers)::int AS other_active_borrowers
                FROM group_books gb
                JOIN group_sizes gs USING (year_level)
                JOIN book_totals bt USING (isbn)
                CROSS JOIN total_students ts
                WHERE gb.borrowers >= 3
                """
            )
            counts.extend(cursor.fetchall())

    rows = []
    popularity_rows = []
    for isbn, borrowers, active in popularity_counts:
        popularity_rows.append((isbn, borrowers, active, (borrowers + 1) / (active + 2)))
    for program, year_level, isbn, borrowers, active, other_borrowers, other_active in counts:
        share = (borrowers + 1) / (active + 2)
        other_share = (other_borrowers + 1) / (other_active + 2)
        lift = share / other_share
        confidence = borrowers / (borrowers + 10)
        rows.append((
            program, year_level, isbn, borrowers, active, other_borrowers,
            other_active, share, other_share, lift, confidence,
            confidence * lift / (1 + lift),
        ))

    with get_search_index_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute("""CREATE TABLE IF NOT EXISTS ml_book_year_stats (
                program TEXT NOT NULL, year_level TEXT NOT NULL, isbn TEXT NOT NULL,
                borrowers INTEGER NOT NULL, active_borrowers INTEGER NOT NULL,
                other_borrowers INTEGER NOT NULL, other_active_borrowers INTEGER NOT NULL,
                share DOUBLE PRECISION NOT NULL, other_share DOUBLE PRECISION NOT NULL,
                lift DOUBLE PRECISION NOT NULL, confidence DOUBLE PRECISION NOT NULL,
                year_relevance DOUBLE PRECISION NOT NULL,
                PRIMARY KEY (program, year_level, isbn))""")
            cursor.execute("""CREATE TABLE IF NOT EXISTS ml_book_popularity (
                isbn TEXT PRIMARY KEY, borrowers INTEGER NOT NULL,
                active_borrowers INTEGER NOT NULL, share DOUBLE PRECISION NOT NULL)""")
            cursor.execute("DELETE FROM ml_book_year_stats")
            cursor.execute("DELETE FROM ml_book_popularity")
            if rows:
                execute_values(
                    cursor,
                    """INSERT INTO ml_book_year_stats
                    (program, year_level, isbn, borrowers, active_borrowers,
                     other_borrowers, other_active_borrowers, share, other_share,
                     lift, confidence, year_relevance) VALUES %s""",
                    rows,
                )
            if popularity_rows:
                execute_values(
                    cursor,
                    "INSERT INTO ml_book_popularity (isbn, borrowers, active_borrowers, share) VALUES %s",
                    popularity_rows,
                )
        connection.commit()
    return len(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Allow catalog and borrowing-statistic refresh when histories are empty.",
    )
    count = snapshot_user_histories(allow_empty=parser.parse_args().allow_empty)
    print(f"Found {count} user histories in the database")
