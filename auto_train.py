from __future__ import annotations

import os
import time
from datetime import datetime, time as datetime_time, timedelta
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from dotenv import load_dotenv
from evaluate_recommender import evaluate_model

from train_svd_recommender import train_model

load_dotenv()


def run():
    enabled = os.getenv("AUTO_TRAIN_ENABLED", "false").strip().lower() in {
        "1",
        "true",
        "yes",
    }
    if not enabled:
        print("Automatic training is disabled (AUTO_TRAIN_ENABLED=false).")
        while True:
            time.sleep(3600)

    schedule_time = os.getenv("AUTO_TRAIN_TIME", "00:00").strip()
    try:
        hour, minute = (int(value) for value in schedule_time.split(":", 1))
        target_time = datetime_time(hour=hour, minute=minute)
    except (ValueError, TypeError):
        raise ValueError("AUTO_TRAIN_TIME must use HH:MM format, for example 00:00.")

    timezone_name = os.getenv("AUTO_TRAIN_TIMEZONE", "UTC").strip()
    try:
        schedule_zone = ZoneInfo(timezone_name)
    except ZoneInfoNotFoundError as error:
        raise ValueError(
            f"AUTO_TRAIN_TIMEZONE is not a valid IANA timezone: {timezone_name}"
        ) from error

    while True:
        now = datetime.now(schedule_zone)
        next_run = datetime.combine(
            now.date(), target_time, tzinfo=schedule_zone
        )
        if next_run <= now:
            next_run += timedelta(days=1)

        sleep_seconds = (next_run - now).total_seconds()
        print(
            f"Next model training: {next_run.isoformat()} "
            f"(timezone={timezone_name})"
        )
        time.sleep(sleep_seconds)

        started = datetime.now(schedule_zone).isoformat()
        print(f"Starting scheduled model training at {started}")
        train_model(refresh_database=True)
        evaluate_model()
        print("Training complete. Waiting for the next midnight schedule.")


if __name__ == "__main__":
    run()
