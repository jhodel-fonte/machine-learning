import argparse
import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from psycopg2.extras import execute_values
from snapshot_user_histories import get_database_connection, snapshot_user_histories
from surprise import Dataset, Reader, SVD, dump
from surprise.model_selection import train_test_split

BASE_DIR = Path(__file__).resolve().parent


def train_model(refresh_database: bool = False):
    load_dotenv(BASE_DIR / ".env")
    if refresh_database:
        print("Refreshing catalog and borrowing statistics from PostgreSQL...")
        snapshot_user_histories()

    print("Loading interactions from PostgreSQL...")
    include_seeded = os.getenv("ML_INCLUDE_SEEDED_DATA", "false").strip().lower() in {
        "1", "true", "yes",
    }
    with get_database_connection() as connection:
        df = pd.read_sql_query(
            """SELECT accounts.ml_user_token AS user_key, btrim(histories.isbn) AS isbn, histories.rating
               FROM user_histories histories JOIN accounts ON accounts.id = histories.user_id
               WHERE accounts.ml_user_token IS NOT NULL AND btrim(accounts.ml_user_token) <> ''
                 AND (histories.is_seeded = FALSE OR %s)
                 AND histories.isbn IS NOT NULL AND btrim(histories.isbn) <> ''
               ORDER BY histories.id""",
            connection,
            params=(include_seeded,),
        )
    if df.empty:
        raise ValueError("Dataset contains no interactions.")

    reader = Reader(rating_scale=(0, 2))
    data = Dataset.load_from_df(df[['user_key', 'isbn', 'rating']], reader)


    print("Splitting data (80/20) and applying random seed 0...")
    trainset, testset = train_test_split(data, test_size=0.20, random_state=0)

    print("Training SVD Model...")
    algo = SVD(random_state=7)
    algo.fit(trainset)

    print("Saving trained model and held-out interactions...")
    dump.dump(str(BASE_DIR / 'library_svd_model.pkl'), algo=algo)
    with get_database_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """CREATE TABLE IF NOT EXISTS ml_recommender_test_interactions (
                       user_key TEXT NOT NULL,
                       isbn TEXT NOT NULL,
                       rating DOUBLE PRECISION NOT NULL
                   )"""
            )
            cursor.execute(
                "CREATE INDEX IF NOT EXISTS ml_recommender_test_user_idx "
                "ON ml_recommender_test_interactions (user_key)"
            )
            cursor.execute("DELETE FROM ml_recommender_test_interactions")
            if testset:
                execute_values(
                    cursor,
                    "INSERT INTO ml_recommender_test_interactions (user_key, isbn, rating) VALUES %s",
                    [(str(uid), str(isbn), float(rating)) for uid, isbn, rating in testset],
                )

    print("Training complete!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--refresh-database",
        action="store_true",
        help="Refresh catalog embeddings and borrowing statistics before training.",
    )
    train_model(refresh_database=parser.parse_args().refresh_database)
