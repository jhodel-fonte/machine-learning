import json
import math
import os
from pathlib import Path

import numpy as np
import psycopg2
from sentence_transformers import SentenceTransformer
from surprise import dump
from dotenv import load_dotenv
from snapshot_user_histories import get_database_connection, get_search_index_connection

BASE_DIR = Path(__file__).resolve().parent
_query_model = None
POPULARITY_PATH = "ml_book_popularity"


def _normalize_isbn(isbn):
    return str(isbn).strip().replace("-", "").replace(" ", "").upper()


def _content_scores(user_key, known_books, history_override=None, include_neighbors=True):
    """Score catalog books against the embedding profile of a user's history."""
    load_dotenv(BASE_DIR / ".env")
    include_seeded = os.getenv("ML_INCLUDE_SEEDED_DATA", "false").strip().lower() in {
        "1", "true", "yes",
    }
    known_books = list({_normalize_isbn(isbn) for isbn in known_books})
    if history_override is None:
        with get_database_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """SELECT histories.isbn, histories.rating FROM user_histories histories
                       JOIN accounts ON accounts.id = histories.user_id
                       WHERE accounts.ml_user_token = %s AND histories.rating > 0
                         AND (histories.is_seeded = FALSE OR %s)
                         AND histories.isbn IS NOT NULL AND btrim(histories.isbn) <> ''""",
                    (user_key, include_seeded),
                )
                history = cursor.fetchall()
    else:
        history = history_override
    if not history:
        return {}

    isbns = list({_normalize_isbn(isbn) for isbn, _ in history})
    with get_search_index_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """SELECT upper(regexp_replace(btrim(isbn), '[-[:space:]]', '', 'g')), embedding::text
                   FROM ml_book_embeddings
                   WHERE upper(regexp_replace(btrim(isbn), '[-[:space:]]', '', 'g')) = ANY(%s)""",
                (isbns,),
            )
            rows = cursor.fetchall()
            vectors = {}
            for isbn, value in rows:
                vectors.setdefault(isbn, []).append(
                    np.fromstring(value.strip("[]"), sep=",", dtype=np.float32)
                )
            vectors = {isbn: np.mean(values, axis=0) for isbn, values in vectors.items()}
            weighted = [vectors[_normalize_isbn(isbn)] * rating for isbn, rating in history
                        if _normalize_isbn(isbn) in vectors]
            if not weighted:
                return {}
            profile = np.mean(weighted, axis=0)
            norm = np.linalg.norm(profile)
            if not norm:
                return {}
            vector_text = "[" + ",".join(f"{value:.8f}" for value in profile / norm) + "]"
            scores = {}
            if include_neighbors:
                cursor.execute(
                    """SELECT isbn, max(1 - distance) FROM (
                           SELECT upper(regexp_replace(btrim(isbn), '[-[:space:]]', '', 'g')) AS isbn,
                                  embedding <=> %s::vector AS distance
                           FROM ml_book_embeddings ORDER BY embedding <=> %s::vector LIMIT 500
                       ) nearest GROUP BY isbn""",
                    (vector_text, vector_text),
                )
                scores = {isbn: float(score) for isbn, score in cursor.fetchall()}
            if known_books:
                cursor.execute(
                    """SELECT upper(regexp_replace(btrim(isbn), '[-[:space:]]', '', 'g')),
                              max(1 - (embedding <=> %s::vector))
                       FROM ml_book_embeddings
                       WHERE upper(regexp_replace(btrim(isbn), '[-[:space:]]', '', 'g')) = ANY(%s)
                       GROUP BY upper(regexp_replace(btrim(isbn), '[-[:space:]]', '', 'g'))""",
                    (vector_text, known_books),
                )
                scores.update({isbn: float(score) for isbn, score in cursor.fetchall()})
            return {isbn: max(0.0, min(1.0, score)) for isbn, score in scores.items()}


def _year_scores(program, year_level):
    if not program or not year_level:
        return {}
    load_dotenv(BASE_DIR / ".env")
    with get_search_index_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """SELECT isbn, year_relevance FROM ml_book_year_stats
                   WHERE program = '*' AND year_level = %s""",
                (year_level.strip().lower(),),
            )
            scores = {isbn: float(score) for isbn, score in cursor.fetchall()}
            cursor.execute(
                """SELECT isbn, year_relevance FROM ml_book_year_stats
                   WHERE program = %s AND year_level = %s""",
                (program.strip().lower(), year_level.strip().lower()),
            )
            scores.update({isbn: float(score) for isbn, score in cursor.fetchall()})
            return scores


def _popularity_scores(isbns):
    if not isbns:
        return {}
    load_dotenv(BASE_DIR / ".env")
    with get_search_index_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                f"SELECT isbn, share FROM {POPULARITY_PATH} WHERE isbn = ANY(%s)",
                (isbns,),
            )
            return {isbn: float(share) for isbn, share in cursor.fetchall()}


def _availability_counts(isbns):
    if not isbns:
        return {}
    load_dotenv(BASE_DIR / ".env")
    with get_database_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """SELECT upper(regexp_replace(btrim(isbn), '[-[:space:]]', '', 'g')),
                          sum(available_copies)::int
                   FROM opac_catalog_view
                   WHERE upper(regexp_replace(btrim(isbn), '[-[:space:]]', '', 'g')) = ANY(%s)
                   GROUP BY upper(regexp_replace(btrim(isbn), '[-[:space:]]', '', 'g'))""",
                (isbns,),
            )
            return {isbn: int(count) for isbn, count in cursor.fetchall()}


def search_books(query, top_n=10):
    global _query_model
    try:
        load_dotenv(BASE_DIR / ".env")
        if _query_model is None:
            _query_model = SentenceTransformer(
                "sentence-transformers/all-MiniLM-L6-v2", device="cpu"
            )
        vector = _query_model.encode(
            [query], normalize_embeddings=True, show_progress_bar=False
        )[0]
        vector_text = "[" + ",".join(f"{value:.8f}" for value in vector) + "]"
        with psycopg2.connect(
            host=os.getenv("ML_INDEX_DB_HOST") or os.getenv("DB_HOST", "localhost"),
            port=os.getenv("ML_INDEX_DB_PORT") or os.getenv("DB_PORT", "5432"),
            dbname=os.getenv("ML_INDEX_DB_DATABASE") or os.environ["DB_DATABASE"],
            user=os.getenv("ML_INDEX_DB_USERNAME") or os.environ["DB_USERNAME"],
            password=os.getenv("ML_INDEX_DB_PASSWORD") or os.environ["DB_PASSWORD"],
            connect_timeout=5,
        ) as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """SELECT isbn, title, 1 - (embedding <=> %s::vector) AS relevance
                       FROM ml_book_embeddings ORDER BY embedding <=> %s::vector LIMIT %s""",
                    (vector_text, vector_text, top_n),
                )
                return [
                    {"isbn": isbn, "title": title, "relevance": round(float(score), 3)}
                    for isbn, title, score in cursor.fetchall()
                ]
    except psycopg2.Error:
        return {"error": "Semantic search index unavailable."}


def get_top_recommendations(
    user_key, borrowed_books, top_n=10, program=None, year_level=None
):
    try:
        _, algo = dump.load(str(BASE_DIR / 'library_svd_model.pkl'))
        with open(BASE_DIR / 'all_books.json', 'r') as f:
            all_books = json.load(f)
    except FileNotFoundError:
        return {"error": "Model files missing. Run training script first."}

    borrowed_books = {_normalize_isbn(book) for book in (borrowed_books or [])}
    try:
        content_scores = _content_scores(user_key, all_books)
    except psycopg2.Error:
        content_scores = {}
    try:
        year_scores = _year_scores(program, year_level)
    except psycopg2.Error:
        year_scores = {}
    books_by_isbn = {}
    for book in [*all_books, *content_scores]:
        books_by_isbn.setdefault(_normalize_isbn(book), book)
    candidates = [book for isbn, book in books_by_isbn.items() if isbn not in borrowed_books]
    isbns = list({_normalize_isbn(book) for book in candidates})
    try:
        popularity = _popularity_scores(isbns)
    except psycopg2.Error:
        popularity = {}
    availability_known = True
    try:
        availability = _availability_counts(isbns)
    except psycopg2.Error:
        availability = {}
        availability_known = False

    recommendations = []
    for book in candidates:
        isbn = _normalize_isbn(book)
        svd_score = max(0.0, min(2.0, float(algo.predict(uid=user_key, iid=book).est)))
        signals = [(svd_score, 0.5)]
        if isbn in content_scores:
            signals.append((2 * content_scores[isbn], 0.3))
        if isbn in year_scores:
            signals.append((2 * year_scores[isbn], 0.2))
        score = sum(value * weight for value, weight in signals) / sum(weight for _, weight in signals)
        available_count = availability.get(isbn, 0)
        recommendations.append({
            "isbn": isbn,
            "predicted_score": round(score, 4),
            "available_copies": available_count if availability_known else None,
            "_availability_sort": available_count,
            "popularity": popularity.get(isbn),
        })

    recommendations.sort(key=lambda item: (-item["predicted_score"], -item["_availability_sort"], item["isbn"]))
    novelty_count = min(3, math.ceil(max(0, top_n) * 0.25))
    if recommendations and novelty_count and popularity:
        shares = sorted(popularity.values())
        median_share = shares[len(shares) // 2]
        best_score = recommendations[0]["predicted_score"]
        regular = [item for item in recommendations if item["popularity"] is None or item["popularity"] > median_share]
        novel = [item for item in recommendations if item["popularity"] is not None
                 and item["popularity"] < median_share
                 and item["predicted_score"] >= best_score - 0.2]
        novel.sort(key=lambda item: (-item["predicted_score"], item["popularity"], -item["_availability_sort"]))
        selected = regular[:max(0, top_n - min(novelty_count, len(novel)))]
        selected.extend(novel[:novelty_count])
        recommendations = selected + [
            item for item in recommendations if item not in selected
        ]

    return [
        {key: value for key, value in item.items() if key not in {"popularity", "_availability_sort"}}
        for item in recommendations[:max(0, top_n)]
    ]
