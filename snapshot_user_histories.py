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

    existing_rows = []
    if OUTPUT_PATH.exists():
        with OUTPUT_PATH.open(newline="", encoding="utf-8") as existing_file:
            reader = csv.DictReader(existing_file)
            expected_fields = ["User-ID", "ISBN", "rating"]
            if reader.fieldnames != expected_fields:
                raise ValueError(
                    f"{OUTPUT_PATH} must contain columns: {', '.join(expected_fields)}"
                )
            existing_rows = [
                (row["User-ID"], row["ISBN"], row["rating"]) for row in reader
            ]

    existing_keys = set(existing_rows)
    new_rows = []
    for user_id, isbn, rating in rows:
        normalized_row = (str(user_id), str(isbn), str(rating))
        if normalized_row not in existing_keys:
            new_rows.append(normalized_row)
            existing_keys.add(normalized_row)
    merged_rows = existing_rows + new_rows

    with tempfile.NamedTemporaryFile(
        mode="w", newline="", encoding="utf-8", dir=OUTPUT_PATH.parent, delete=False
    ) as temporary_file:
        writer = csv.writer(temporary_file)
        writer.writerow(["User-ID", "ISBN", "rating"])
        writer.writerows(merged_rows)
        temporary_path = Path(temporary_file.name)

    temporary_path.replace(OUTPUT_PATH)
    return len(new_rows)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-empty",
        action="store_true",
        help="Write an empty CSV when user_histories has no rows.",
    )
    count = snapshot_user_histories(allow_empty=parser.parse_args().allow_empty)
    print(f"Added {count} new user histories to {OUTPUT_PATH}")
