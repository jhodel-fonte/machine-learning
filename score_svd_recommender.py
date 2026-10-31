import json
import os
from pathlib import Path

import psycopg2
from sentence_transformers import SentenceTransformer
from surprise import dump
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
_query_model = None


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

    borrowed_books = {str(book).strip() for book in (borrowed_books or [])}
    try:
        with open(BASE_DIR / "cohort_preferences.json", encoding="utf-8") as f:
            cohort_preferences = json.load(f)
    except FileNotFoundError:
        cohort_preferences = []

    cohort_key = (str(program or "").strip().lower(), str(year_level or "").strip().lower())
    cohort_scores = {
        item["isbn"]: item["mean_rating"]
        for item in cohort_preferences
        if (item["program"], item["year_level"]) == cohort_key
    }
    recommendations = []
    for book in all_books:
        if str(book) not in borrowed_books:
            predicted = algo.predict(uid=user_key, iid=book).est
            cohort = cohort_scores.get(str(book))
            score = 0.7 * predicted + 0.3 * cohort if cohort is not None else predicted
            recommendations.append({"isbn": book, "predicted_score": round(score, 2)})

    recommendations.sort(key=lambda item: item["predicted_score"], reverse=True)
    return recommendations[:max(0, top_n)]
