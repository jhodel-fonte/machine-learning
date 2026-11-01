"""Delete one explicitly selected set of ML data after confirmation."""

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parent
FILES = {
    "model": ["library_svd_model.pkl"],
    "metrics": ["model_metrics.json", "model_metrics_history.jsonl"],
}
DATABASE_ACTIONS = {
    "test-interactions": ("main", ["ml_recommender_test_interactions"], ""),
    "index": ("index", ["ml_book_embeddings", "ml_book_year_stats", "ml_book_popularity"], ""),
    "seeded-histories": ("main", ["user_histories"], " WHERE is_seeded = TRUE"),
    "all-histories": ("main", ["user_histories"], ""),
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=[*FILES, *DATABASE_ACTIONS])
    action = parser.parse_args().action
    if action in FILES:
        print("Files to delete:", ", ".join(str(ROOT / name) for name in FILES[action]))
        if input('Type DELETE to confirm: ').strip() != 'DELETE':
            print('Cancelled.')
            return
        for name in FILES[action]:
            (ROOT / name).unlink(missing_ok=True)
        print('Selected files deleted if present.')
        return

    from dotenv import load_dotenv
    from snapshot_user_histories import get_database_connection, get_search_index_connection

    load_dotenv(ROOT / '.env')
    database, tables, condition = DATABASE_ACTIONS[action]
    connect = get_search_index_connection if database == 'index' else get_database_connection
    with connect() as connection:
        with connection.cursor() as cursor:
            cursor.execute('SELECT current_database(), current_user, inet_server_addr(), inet_server_port()')
            print('Connected database, user, server address and port:', cursor.fetchone())
            print('Rows to delete from:', ', '.join(tables))
            if action == 'seeded-histories':
                print('Only rows marked as seeded will be deleted.')
            if action == 'all-histories':
                print('WARNING: This permanently deletes all student interactions, including real history.')
            if input('Type DELETE to confirm: ').strip() != 'DELETE':
                print('Cancelled.')
                return
            from psycopg2 import sql
            for table in tables:
                cursor.execute('SELECT to_regclass(%s)', (table,))
                if cursor.fetchone()[0] is None:
                    print(f'{table}: table does not exist; skipped.')
                    continue
                cursor.execute(sql.SQL('DELETE FROM {}' + condition).format(sql.Identifier(table)))
                print(f'{table}: deleted {cursor.rowcount} rows.')


if __name__ == '__main__':
    main()
