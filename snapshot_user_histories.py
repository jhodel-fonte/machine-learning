from __future__ import annotations

import csv
import argparse
import os
import tempfile
from pathlib import Path

import psycopg2
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = BASE_DIR / "dataset" / "interactions.csv"


def get_database_connection():
    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        dbname=os.environ["DB_DATABASE"],
        user=os.environ["DB_USERNAME"],
        password=os.environ["DB_PASSWORD"],
        connect_timeout=5,
    )


def snapshot_user_histories(allow_empty: bool = False) -> int:
    load_dotenv(BASE_DIR / ".env")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with get_database_connection() as connection:
        with connection.cursor() as cursor:
            cursor.execute(
                """
                SELECT encrypter_id, isbn, rating
                FROM user_histories
                WHERE encrypter_id IS NOT NULL
                  AND btrim(encrypter_id) <> ''
                ORDER BY id
                """
            )
            rows = cursor.fetchall()

    if not rows and not allow_empty:
        raise RuntimeError(
            "user_histories returned 0 rows; existing interactions.csv was left unchanged. "
            "Use --allow-empty only if an empty training dataset is intentional."
        )

    with tempfile.NamedTemporaryFile(
        mode="w", newline="", encoding="utf-8", dir=OUTPUT_PATH.parent, delete=False
    ) as temporary_file:
        writer = csv.writer(temporary_file)
        writer.writerow(["User-ID", "ISBN", "rating"])
        writer.writerows(rows)
        temporary_path = Path(temporary_file.name)

    temporary_path.replace(OUTPUT_PATH)
    return len(rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Write an empty CSV when user_histories has no rows.",
    )
    count = snapshot_user_histories(allow_empty=parser.parse_args().allow_empty)
    print(f"Saved {count} user histories to {OUTPUT_PATH}")
