import argparse
import json
import math
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
from snapshot_user_histories import (
    BASE_DIR,
    get_database_connection,
    get_search_index_connection,
)

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
        if not hasattr(prediction, "est"):
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


def evaluate_recommendation_strategies(algo, predictions, top_n=10, catalog_from_database=False):
    """Evaluate standalone and serving-style top-N rankings on the full catalog."""
    load_dotenv(BASE_DIR / ".env")
    trainset = algo.trainset
    try:
        with (get_database_connection() if catalog_from_database else get_search_index_connection()) as connection:
            with connection.cursor() as cursor:
                cursor.execute("""SELECT bd.isbn FROM book_details bd
                    JOIN book_datas data ON data.id = bd.book_data_id
                    WHERE bd.isbn IS NOT NULL AND btrim(bd.isbn) <> ''
                      AND bd.deleted_at IS NULL AND data.deleted_at IS NULL""" if catalog_from_database else
                    """SELECT isbn FROM ml_book_embeddings
                       WHERE isbn IS NOT NULL AND btrim(isbn) <> ''"""
                )
                catalog_rows = cursor.fetchall()
    except psycopg2.Error:
        if catalog_from_database:
            raise
        catalog_rows = []
    catalog = {}
    for (raw_isbn,) in catalog_rows:
        catalog.setdefault(_normalize_isbn(raw_isbn), raw_isbn.strip())
    catalog_source = "active library catalog" if catalog_from_database else "ml_book_embeddings"
    if not catalog:
        if catalog_from_database:
            raise ValueError("No active catalog ISBNs found; evaluation was not saved.")
        catalog = {
            _normalize_isbn(trainset.to_raw_iid(inner_iid)): str(trainset.to_raw_iid(inner_iid)).strip()
            for inner_iid in range(trainset.n_items)
        }
        catalog_source = "training_interactions fallback"

    train_histories = {}
    train_seen = {}
    borrower_users = {}
    for inner_uid, ratings in trainset.ur.items():
        uid = str(trainset.to_raw_uid(inner_uid))
        history = [(trainset.to_raw_iid(inner_iid), rating) for inner_iid, rating in ratings]
        train_histories[uid] = history
        train_seen[uid] = {_normalize_isbn(isbn) for isbn, _ in history}
        for isbn, rating in history:
            if rating >= 2:
                borrower_users.setdefault(_normalize_isbn(isbn), set()).add(uid)

    all_borrowers = set().union(*borrower_users.values()) if borrower_users else set()
    popularity = {
        isbn: (len(users) + 1) / (len(all_borrowers) + 2)
        for isbn, users in borrower_users.items()
    }

    all_users = set(train_histories)
    test_users = {str(prediction.uid) for prediction in predictions}
    contexts = {}
    try:
        with get_database_connection() as connection:
            with connection.cursor() as cursor:
                cursor.execute(
                    """SELECT accounts.ml_user_token, students.program, students.year_level
                       FROM accounts JOIN students ON students.account_id = accounts.id
                       WHERE accounts.ml_user_token = ANY(%s)""",
                    (list(all_users | test_users),),
                )
                contexts = {
                    str(uid): (
                        program.strip().lower() if program else None,
                        year.strip().lower() if year else None,
                    )
                    for uid, program, year in cursor.fetchall()
                }
    except psycopg2.Error:
        pass

    # Build smoothed cohort scores from training borrowers only.
    cohort_active = {}
    cohort_books = {}
    for uid, history in train_histories.items():
        program, year = contexts.get(uid, (None, None))
        if not year:
            continue
        groups = [("*", year)]
        if program:
            groups.append((program, year))
        borrowed = {_normalize_isbn(isbn) for isbn, rating in history if rating >= 2}
        for group in groups:
            cohort_active.setdefault(group, set()).update([uid] if borrowed else [])
            for isbn in borrowed:
                cohort_books.setdefault(group, {}).setdefault(isbn, set()).add(uid)

    cohort_scores = {}
    for group, books in cohort_books.items():
        active = cohort_active[group]
        scores = {}
        for isbn, borrowers in books.items():
            if len(borrowers) < 3:
                continue
            other_borrowers = len(borrower_users.get(isbn, set()) - borrowers)
            other_active = len(all_borrowers - active)
            share = (len(borrowers) + 1) / (len(active) + 2)
            other_share = (other_borrowers + 1) / (other_active + 2)
            lift = share / other_share
            confidence = len(borrowers) / (len(borrowers) + 10)
            scores[isbn] = confidence * lift / (1 + lift)
        cohort_scores[group] = scores

    try:
        availability = _availability_counts(list(catalog))
        availability_available = True
    except psycopg2.Error:
        availability = {}
        availability_available = False

    targets = {}
    for prediction in predictions:
        uid = str(prediction.uid)
        isbn = _normalize_isbn(prediction.iid)
        if isbn in catalog and isbn not in train_seen.get(uid, set()):
            targets.setdefault(uid, {})[isbn] = max(
                float(prediction.r_ui), targets.get(uid, {}).get(isbn, 0.0)
            )

    strategy_names = (
        "svd_only", "content_only", "program_year_only", "popularity_only",
        "hybrid_score", "hybrid_with_availability", "hybrid_availability_novelty",
    )
    ranked_by_strategy = {name: {} for name in strategy_names}
    popularity_by_user = {}
    users_evaluated = 0
    target_count = 0
    for uid, relevant in targets.items():
        if not relevant:
            continue
        users_evaluated += 1
        target_count += len(relevant)
        known = train_seen.get(uid, set())
        candidates = {isbn: raw for isbn, raw in catalog.items() if isbn not in known}
        popularity_by_user[uid] = [popularity[isbn] for isbn in candidates if isbn in popularity]
        try:
            content = _content_scores(
                uid, list(candidates.values()),
                history_override=train_histories.get(uid, []), include_neighbors=False,
            )
        except psycopg2.Error:
            content = {}
        program, year = contexts.get(uid, (None, None))
        year_scores = dict(cohort_scores.get(("*", year), {})) if year else {}
        if program and year:
            year_scores.update(cohort_scores.get((program, year), {}))

        candidates_by_strategy = {name: [] for name in strategy_names}
        for isbn, raw_isbn in candidates.items():
            if not raw_isbn:
                continue
            svd_score = max(0.0, min(2.0, float(algo.predict(uid, raw_isbn).est)))
            content_score = content.get(isbn)
            cohort_score = year_scores.get(isbn)
            signals = [(svd_score, 0.5)]
            if content_score is not None:
                signals.append((2 * content_score, 0.3))
            if cohort_score is not None:
                signals.append((2 * cohort_score, 0.2))
            hybrid_score = sum(score * weight for score, weight in signals) / sum(
                weight for _, weight in signals
            )
            item = {
                "isbn": isbn,
                "score": 0.0,
                "popularity": popularity.get(isbn),
                "available": availability.get(isbn, 0),
            }
            candidates_by_strategy["svd_only"].append({**item, "score": svd_score})
            if content_score is not None:
                candidates_by_strategy["content_only"].append({**item, "score": content_score})
            if cohort_score is not None:
                candidates_by_strategy["program_year_only"].append({**item, "score": cohort_score})
            if isbn in popularity:
                candidates_by_strategy["popularity_only"].append({**item, "score": popularity[isbn]})
            candidates_by_strategy["hybrid_score"].append({**item, "score": hybrid_score})
            candidates_by_strategy["hybrid_with_availability"].append({**item, "score": hybrid_score})
            candidates_by_strategy["hybrid_availability_novelty"].append({**item, "score": hybrid_score})

        for name, items in candidates_by_strategy.items():
            items.sort(key=lambda item: (-item["score"], item["isbn"]))
            if name in {"hybrid_with_availability", "hybrid_availability_novelty"}:
                items.sort(key=lambda item: (-item["score"], -item["available"], item["isbn"]))
            if name == "hybrid_availability_novelty" and items:
                shares = sorted(item["popularity"] for item in items if item["popularity"] is not None)
                if shares:
                    median_share = shares[len(shares) // 2]
                    novel_count = min(3, math.ceil(top_n * 0.25))
                    best_score = items[0]["score"]
                    regular = [item for item in items if item["popularity"] is None or item["popularity"] > median_share]
                    novel = [item for item in items if item["popularity"] is not None
                             and item["popularity"] < median_share
                             and item["score"] >= best_score - 0.2]
                    novel.sort(key=lambda item: (-item["score"], item["popularity"], -item["available"]))
                    selected = regular[:max(0, top_n - min(novel_count, len(novel)))]
                    selected.extend(novel[:novel_count])
                    items = selected + [item for item in items if item not in selected]
            ranked_by_strategy[name][uid] = items[:top_n]

    def ranking_metrics(strategy):
        precisions = []
        recalls = []
        ndcgs = []
        recommended = set()
        item_count = available_count = novel_count = popularity_count = 0
        for uid, relevant in targets.items():
            if not relevant:
                continue
            items = ranked_by_strategy[strategy].get(uid, [])
            recommended.update(item["isbn"] for item in items)
            item_count += len(items)
            hits = sum(item["isbn"] in relevant for item in items)
            precisions.append(hits / min(top_n, len(items)) if items else 0.0)
            recalls.append(hits / len(relevant))
            dcg = sum(
                (2 ** relevant[item["isbn"]] - 1) / math.log2(index + 2)
                for index, item in enumerate(items) if item["isbn"] in relevant
            )
            ideal = sorted(relevant.values(), reverse=True)[:top_n]
            ideal_dcg = sum((2 ** rating - 1) / math.log2(index + 2) for index, rating in enumerate(ideal))
            ndcgs.append(dcg / ideal_dcg if ideal_dcg else 0.0)
            available_count += sum(item["available"] > 0 for item in items)
            known_popularity = [item for item in items if item["popularity"] is not None]
            if known_popularity:
                popularity_values = popularity_by_user.get(uid, [])
                median_share = sorted(popularity_values)[len(popularity_values) // 2] if popularity_values else None
                if median_share is not None:
                    popularity_count += len(known_popularity)
                    novel_count += sum(item["popularity"] < median_share for item in known_popularity)

        users = len(recalls)
        return {
            "users_evaluated": users,
            "held_out_relevant_items": sum(len(items) for items in targets.values()),
            "recommended_items": item_count,
            "precision_at_10": round(float(np.mean(precisions)), 4) if users else None,
            "recall_at_10": round(float(np.mean(recalls)), 4) if users else None,
            "ndcg_at_10": round(float(np.mean(ndcgs)), 4) if users else None,
            "catalog_coverage": round(len(recommended) / len(catalog), 4) if catalog else None,
            "availability_rate_at_10": (
                round(available_count / item_count, 4) if item_count and availability_available else None
            ),
            "novel_item_rate_at_10": round(novel_count / popularity_count, 4) if popularity_count else None,
        }

    result = {
        name: ranking_metrics(name) for name in strategy_names
    }
    return {
        "candidate_catalog_books": len(catalog),
        "users_with_rankable_test_items": users_evaluated,
        "held_out_relevant_items": target_count,
        "strategies": result,
        "catalog_source": catalog_source,
        "method": ("Catalog ISBNs were read directly from the active library database. The existing model and held-out data are used; they may include seeded data. Content scores require the separate search index. " if catalog_from_database else "") + "Each strategy ranks the candidate catalog after excluding training-seen books; this is an offline ranking, not the online candidate-generation list. Test interactions are treated as relevant; no impression or unborrowed-negative data is available. Popularity and cohort scores use training interactions only; the split remains random, and current student program/year is used.",
    }


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


def get_testset_from_database():
    load_dotenv(BASE_DIR / ".env")
    with get_database_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """SELECT user_key, isbn, rating
                   FROM ml_recommender_test_interactions
                   ORDER BY user_key, isbn"""
            )
            return cursor.fetchall()


def evaluate_model(catalog_from_database=False):
    print("Loading saved model and test data...")
    try:
        _, algo = dump.load(str(BASE_DIR / 'library_svd_model.pkl'))
    except FileNotFoundError:
        print("Error: Run train_svd_recommender.py first.")
        return False

    try:
        testset = get_testset_from_database()
    except psycopg2.Error as error:
        print(f"Error: Could not load the held-out interactions from PostgreSQL: {error}")
        return False
    if not testset:
        print("Error: No held-out interactions found. Run train_svd_recommender.py first.")
        return False

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
        "recommendation_evaluation": evaluate_recommendation_strategies(algo, predictions, catalog_from_database=catalog_from_database),
        "conclusion": build_conclusion(mae, ndcg_statistics["mean"] or 0.0),
    }
    if catalog_from_database:
        metrics['ndcg_at_10'] = None  # The held-out-only score is not a catalog ranking.
        metrics.pop('ndcg_statistics')
        if not metrics['signal_evaluation']['coverage']['content_scored_interactions']:
            strategies = metrics['recommendation_evaluation']['strategies']
            for name in ('content_only', 'hybrid_score', 'hybrid_with_availability', 'hybrid_availability_novelty'):
                strategies.pop(name, None)
            metrics['recommendation_evaluation']['method'] += ' Content and hybrid comparisons are omitted because content scores were unavailable.'
        strategies = metrics['recommendation_evaluation']['strategies']
        for name in ('popularity_only', 'program_year_only'):
            result = strategies[name]
            if result['recommended_items'] < 10 * result['users_evaluated']:
                strategies.pop(name)
                metrics['recommendation_evaluation']['method'] += f' {name} is omitted because it did not produce 10 items per user on average.'
        metrics.pop('signal_evaluation')
        metrics['conclusion'] = 'Offline catalog ranking using the existing model and held-out interactions; this is not an online recommendation test.'
    
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
    return True

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--catalog-from-database', action='store_true', help='Evaluate against active library catalog ISBNs.')
    if not evaluate_model(catalog_from_database=parser.parse_args().catalog_from_database):
        raise SystemExit(1)
