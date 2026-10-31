import json
import math
import pickle
from datetime import datetime, timezone
import numpy as np
import psycopg2
from dotenv import load_dotenv
from surprise import accuracy, dump
from score_svd_recommender import (
    _availability_counts,
    _content_scores,
    _normalize_isbn,
    _popularity_scores,
    _year_scores,
)
from snapshot_user_histories import BASE_DIR, get_database_connection

METRICS_PATH = BASE_DIR / "model_metrics.json"
METRICS_HISTORY_PATH = BASE_DIR / "model_metrics_history.jsonl"


def describe(values):
    numbers = [float(value) for value in values]
    if not numbers:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "min": None,
            "max": None,
            "standard_deviation": None,
        }

    return {
        "count": len(numbers),
        "mean": round(float(np.mean(numbers)), 4),
        "median": round(float(np.median(numbers)), 4),
        "min": round(min(numbers), 4),
        "max": round(max(numbers), 4),
        "standard_deviation": round(float(np.std(numbers)), 4),
    }


def calculate_ndcg(predictions, top_n=10):
    by_user = {}
    for prediction in predictions:
        if isinstance(prediction, tuple):
            uid, actual, score = prediction
            by_user.setdefault(str(uid), []).append((float(actual), float(score)))
        else:
            by_user.setdefault(str(prediction.uid), []).append(
                (float(prediction.r_ui), float(prediction.est))
            )

    scores = []
    for user_predictions in by_user.values():
        ranked = sorted(user_predictions, key=lambda item: item[1], reverse=True)[:top_n]
        ideal = sorted(user_predictions, key=lambda item: item[0], reverse=True)[:top_n]

        def dcg(items):
            return sum(
                (2 ** item[0] - 1) / math.log2(index + 2)
                for index, item in enumerate(items)
            )

        ideal_dcg = dcg(ideal)
        scores.append(dcg(ranked) / ideal_dcg if ideal_dcg else 0.0)

    return scores


def evaluate_signals(algo, predictions):
    """Compare serving signals on the same held-out interactions."""
    load_dotenv(BASE_DIR / ".env")
    users = {str(prediction.uid) for prediction in predictions}
    contexts = {}
    try:
        with get_database_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """SELECT accounts.ml_user_token, students.program, students.year_level
                       FROM accounts JOIN students ON students.account_id = accounts.id
                       WHERE accounts.ml_user_token = ANY(%s)""",
                    (list(users),),
                )
                contexts = {str(uid): (program, year) for uid, program, year in cursor.fetchall()}
    except psycopg2.Error:
        pass

    by_user = {}
    for prediction in predictions:
        by_user.setdefault(str(prediction.uid), []).append(prediction)
    rows = {name: [] for name in ("svd", "content", "year_program", "hybrid")}
    isbn_values = {_normalize_isbn(prediction.iid) for prediction in predictions}
    try:
        popularity = _popularity_scores(list(isbn_values))
    except psycopg2.Error:
        popularity = {}
    try:
        availability = _availability_counts(list(isbn_values))
    except psycopg2.Error:
        availability = {}

    content_count = year_count = 0
    for uid, user_predictions in by_user.items():
        try:
            inner_uid = algo.trainset.to_inner_uid(uid)
            history = [
                (algo.trainset.to_raw_iid(inner_iid), rating)
                for inner_iid, rating in algo.trainset.ur[inner_uid]
                if rating > 0
            ]
            content = _content_scores(
                uid, [prediction.iid for prediction in user_predictions],
                history_override=history, include_neighbors=False,
            )
        except (KeyError, ValueError, psycopg2.Error):
            content = {}
        program, year = contexts.get(uid, (None, None))
        try:
            year_scores = _year_scores(program, year)
        except psycopg2.Error:
            year_scores = {}

        for prediction in user_predictions:
            isbn = _normalize_isbn(prediction.iid)
            svd_score = max(0.0, min(2.0, float(prediction.est)))
            has_content = isbn in content
            has_year = isbn in year_scores
            content_score = 2 * content.get(isbn, 0.0)
            year_score = 2 * year_scores.get(isbn, 0.0)
            signals = [(svd_score, 0.5)]
            if has_content:
                signals.append((content_score, 0.3))
                content_count += 1
            if has_year:
                signals.append((year_score, 0.2))
                year_count += 1
            combined = sum(score * weight for score, weight in signals) / sum(
                weight for _, weight in signals
            )
            rows["svd"].append((uid, prediction.r_ui, svd_score))
            rows["hybrid"].append((uid, prediction.r_ui, combined))
            if has_content:
                rows["content"].append((uid, prediction.r_ui, content_score))
            if has_year:
                rows["year_program"].append((uid, prediction.r_ui, year_score))

    def metrics_for(signal_rows):
        errors = [actual - score for _, actual, score in signal_rows]
        ndcg = describe(calculate_ndcg(signal_rows))
        return {
            "interactions": len(signal_rows),
            "rmse": round(float(np.sqrt(np.mean(np.square(errors)))), 4) if errors else None,
            "mae": round(float(np.mean(np.abs(errors))), 4) if errors else None,
            "ndcg_at_10": ndcg["mean"],
        }

    total = len(predictions)
    result = {
        name: metrics_for(signal_rows) for name, signal_rows in rows.items()
    }
    result.update({
        "coverage": {
            "content_scored_interactions": content_count,
            "year_program_scored_interactions": year_count,
            "popularity_scored_books": len(popularity),
            "availability_listed_books": len(availability),
            "candidate_books": len(isbn_values),
            "test_interactions": total,
        },
        "method": "Random held-out interactions; no impression or unborrowed-negative data. Signal metrics are diagnostic, not online ranking results.",
    })
    return result


def build_conclusion(mae, ndcg_mean):
    if mae <= 0.5:
        error_conclusion = "Prediction error is low for the 0-to-2 interaction scale."
    elif mae <= 1.0:
        error_conclusion = "Prediction error is moderate for the 0-to-2 interaction scale."
    else:
        error_conclusion = "Prediction error is high for the 0-to-2 interaction scale."

    if ndcg_mean >= 0.8:
        ranking_conclusion = "The model ranks relevant items well in the held-out test data."
    elif ndcg_mean >= 0.5:
        ranking_conclusion = "The model provides moderate ranking quality in the held-out test data."
    else:
        ranking_conclusion = "The model ranking quality needs improvement."

    return f"{error_conclusion} {ranking_conclusion}"


def get_training_data_summary(algo):
    return {
        "training_interactions": algo.trainset.n_ratings,
        "training_users": len(algo.trainset.ur),
    }


def evaluate_model():
    print("Loading saved model and test data...")
    try:
        _, algo = dump.load(str(BASE_DIR / 'library_svd_model.pkl'))
        with open(BASE_DIR / 'testset.pkl', 'rb') as f:
            testset = pickle.load(f)
    except FileNotFoundError:
        print("Error: Run train_svd_recommender.py first.")
        return

    print("Evaluating test predictions...")
    predictions = algo.test(testset)
    
    rmse = accuracy.rmse(predictions, verbose=False)
    mae = accuracy.mae(predictions, verbose=False)
    absolute_errors = [abs(prediction.r_ui - prediction.est) for prediction in predictions]
    actual_ratings = [prediction.r_ui for prediction in predictions]
    predicted_ratings = [prediction.est for prediction in predictions]
    ndcg_scores = calculate_ndcg(predictions)
    ndcg_statistics = describe(ndcg_scores)
    training_data = get_training_data_summary(algo)

    metrics = {
        "evaluated_at": datetime.now(timezone.utc).isoformat(),
        "training_data": training_data,
        "evaluation": {
            "test_interactions": len(predictions),
            "test_users": len({str(prediction.uid) for prediction in predictions}),
            "rating_scale": [0, 2],
        },
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "actual_rating_statistics": describe(actual_ratings),
        "predicted_rating_statistics": describe(predicted_ratings),
        "absolute_error_statistics": describe(absolute_errors),
        "ndcg_at_10": ndcg_statistics["mean"],
        "ndcg_statistics": ndcg_statistics,
        "signal_evaluation": evaluate_signals(algo, predictions),
        "conclusion": build_conclusion(mae, ndcg_statistics["mean"] or 0.0),
    }
    
    with open(METRICS_PATH, 'w', encoding='utf-8') as f:
        json.dump(metrics, f, indent=2)

    # Keep an append-only record so later evaluations never erase prior results.
    with open(METRICS_HISTORY_PATH, 'a', encoding='utf-8') as history_file:
        history_file.write(json.dumps(metrics, separators=(",", ":")) + "\n")
        
    print(
        "Metrics saved to model_metrics.json -> "
        f"RMSE: {metrics['rmse']}, MAE: {metrics['mae']}, "
        f"NDCG@10: {metrics['ndcg_at_10']}"
    )

if __name__ == '__main__':
    evaluate_model()
