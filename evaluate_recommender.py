import csv
import json
import math
import pickle
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from surprise import accuracy, dump

BASE_DIR = Path(__file__).resolve().parent
METRICS_PATH = BASE_DIR / "model_metrics.json"
METRICS_HISTORY_PATH = BASE_DIR / "model_metrics_history.jsonl"
DATASET_PATH = BASE_DIR / "dataset" / "interactions.csv"


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
        by_user.setdefault(str(prediction.uid), []).append(prediction)

    scores = []
    for user_predictions in by_user.values():
        ranked = sorted(user_predictions, key=lambda item: item.est, reverse=True)[:top_n]
        ideal = sorted(user_predictions, key=lambda item: item.r_ui, reverse=True)[:top_n]

        def dcg(items):
            return sum(
                (2 ** float(item.r_ui) - 1) / math.log2(index + 2)
                for index, item in enumerate(items)
            )

        ideal_dcg = dcg(ideal)
        scores.append(dcg(ranked) / ideal_dcg if ideal_dcg else 0.0)

    return scores


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


def get_training_data_summary():
    with DATASET_PATH.open(newline="", encoding="utf-8") as dataset_file:
        reader = csv.DictReader(dataset_file)
        interaction_count = 0
        users = set()
        for row in reader:
            interaction_count += 1
            users.add(row["User-Key"])

    return {
        "training_interactions": interaction_count,
        "training_users": len(users),
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
    training_data = get_training_data_summary()

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
