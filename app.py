from flask import Flask, request, jsonify
import os
import json
import hmac
import threading
import time
from contextlib import closing

import psycopg2
from dotenv import load_dotenv
from werkzeug.exceptions import RequestEntityTooLarge

from score_svd_recommender import get_top_recommendations
from evaluate_recommender import METRICS_HISTORY_PATH

load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = int(os.getenv("MAX_REQUEST_BYTES", "16384"))

_rate_limit_lock = threading.Lock()
_request_timestamps = {}


def _api_key_is_valid():
    configured_key = os.getenv("API_KEY") or os.getenv("ML_API_KEY")
    supplied_key = request.headers.get("X-API-Key")
    return bool(
        configured_key
        and supplied_key
        and hmac.compare_digest(supplied_key, configured_key)
    )


def _rate_limit_exceeded(client_key):
    limit = max(1, int(os.getenv("RATE_LIMIT_REQUESTS", "60")))
    window = max(1, int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60")))
    now = time.monotonic()

    with _rate_limit_lock:
        timestamps = [
            timestamp
            for timestamp in _request_timestamps.get(client_key, [])
            if now - timestamp < window
        ]
        if len(timestamps) >= limit:
            _request_timestamps[client_key] = timestamps
            return True, max(1, int(window - (now - timestamps[0])))
        timestamps.append(now)
        _request_timestamps[client_key] = timestamps
        return False, 0


@app.before_request
def protect_api():
    if not request.path.startswith("/api/"):
        return None

    if not _api_key_is_valid():
        return jsonify({"error": "Valid X-API-Key header is required"}), 401

    client_key = request.headers.get("X-API-Key")
    exceeded, retry_after = _rate_limit_exceeded(client_key)
    if exceeded:
        response = jsonify({"error": "Too many requests"})
        response.status_code = 429
        response.headers["Retry-After"] = str(retry_after)
        return response
    return None


@app.errorhandler(RequestEntityTooLarge)
def request_too_large(_error):
    return jsonify({"error": "Request body is too large"}), 413


@app.route('/health', methods=['GET'])
def health():
    return jsonify({"status": "ok"}), 200


def get_database_connection():
    return psycopg2.connect(
        host=os.getenv('DB_HOST', 'localhost'),
        port=os.getenv('DB_PORT', '5432'),
        dbname=os.getenv('DB_DATABASE'),
        user=os.getenv('DB_USERNAME'),
        password=os.getenv('DB_PASSWORD'),
        connect_timeout=5,
    )


@app.route('/api/db/health', methods=['GET'])
def database_health():
    try:
        with closing(get_database_connection()) as connection:
            with connection.cursor() as cursor:
                cursor.execute('SELECT 1')
        return jsonify({"status": "ok", "database": os.getenv('DB_DATABASE')}), 200
    except psycopg2.Error as error:
        app.logger.error("Database health check failed: %s", error)
        return jsonify({"status": "error", "error": "Database connection failed"}), 503


@app.route('/api/recommend', methods=['POST'])
def recommend():
    data = request.get_json(silent=True) or {}
    encrypter_id = data.get('encrypter_id')
    borrowed_books = data.get('borrowed_books', []) 

    if not encrypter_id:
        return jsonify({"error": "encrypter_id is required"}), 400
    if not isinstance(borrowed_books, list):
        return jsonify({"error": "borrowed_books must be an array"}), 400

    # Call the scoring script
    top_10 = get_top_recommendations(encrypter_id, borrowed_books, top_n=10)
    
    if isinstance(top_10, dict) and "error" in top_10:
        return jsonify(top_10), 500

    return jsonify({
        "encrypter_id": encrypter_id,
        "recommendations": top_10
    }), 200

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    # Return metrics for the Laravel admin dashboard
    if os.path.exists('model_metrics.json'):
        with open('model_metrics.json', 'r') as f:
            metrics = json.load(f)
        return jsonify(metrics), 200
    
    return jsonify({"error": "Metrics not found. Run evaluate_recommender.py first."}), 404


@app.route('/api/metrics/history', methods=['GET'])
def get_metrics_history():
    if not METRICS_HISTORY_PATH.exists():
        return jsonify({"history": []}), 200

    history = []
    with METRICS_HISTORY_PATH.open(encoding='utf-8') as history_file:
        for line in history_file:
            if line.strip():
                history.append(json.loads(line))

    return jsonify({"history": list(reversed(history))}), 200

if __name__ == '__main__':
    # Listen on all interfaces (0.0.0.0) so Docker can map the port
    app.run(host='0.0.0.0', port=5000, debug=True)