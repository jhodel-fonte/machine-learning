# machine-learning

## Ubuntu Azure VM deployment (native Python, no Docker)

Laravel and this ML API run directly on the same Ubuntu VM. Systemd keeps the
API and scheduled trainer running as the restricted `pgpc-ml` account. The API
listens only on `127.0.0.1:8005`, which Laravel can call locally; do not open
port 8005 in the Azure Network Security Group.

Clone this repository to `/opt/machine_learning`, then configure its secrets
and database connections:

```bash
sudo git clone <repository-url> /opt/machine_learning
cd /opt/machine_learning
sudo cp .env.example .env
sudo nano .env
```

Set `DB_*` to the main PostgreSQL database used by Laravel and set `API_KEY`
to a long random secret. The service runs natively, so use the database's real
hostname (for PostgreSQL on this same VM, `DB_HOST=localhost`). To keep the
large book search index off the main database, set all `ML_INDEX_DB_*` values
to the separate PostgreSQL database. It needs pgvector installed, and the ML
account needs permission to create/update the index table and index. Leave
those fields blank only if you intend the index to share the main database.

In the Laravel `.env` on the VM, set the same API key and local URL, then run
the token migration before the ML snapshot:

```env
ML_RECOMMENDER_URL=http://127.0.0.1:8005
ML_API_KEY=<same-long-secret-as-API_KEY>
```

```bash
cd /path/to/pgpc-lib
php artisan migrate
php artisan config:clear
```

You do not need to stop Docker or other VM services to run this native ML
service. Stop only an old ML service if it is already occupying port 8005.

Install the Python environment, download the embedding model, refresh the
catalog and interaction data, train/evaluate the recommender, and start the
systemd services with one command:

```bash
sudo bash deploy/install-ubuntu.sh
```

The first catalog indexing run can take a while and needs enough disk space
for Python packages, the embedding model cache, and database index growth.
The script writes model artifacts under `/opt/machine_learning`; that directory
must remain in place for the API and scheduled trainer.

Check the API and service logs:

```bash
curl -fsS http://127.0.0.1:8005/health
sudo systemctl status pgpc-ml-api pgpc-ml-trainer
sudo journalctl -u pgpc-ml-api -n 100 --no-pager
```

Automatic retraining stays idle until enabled. To train every day at midnight
Philippine time, set these values in `/opt/machine_learning/.env`, then restart
the trainer:

```env
AUTO_TRAIN_ENABLED=true
AUTO_TRAIN_TIME=00:00
AUTO_TRAIN_TIMEZONE=Asia/Manila
```

```bash
sudo systemctl restart pgpc-ml-trainer
```

Keep PostgreSQL private. For an Azure PostgreSQL server, allow its firewall
connection from the VM and use the server hostname in `.env`; do not expose
the database to the public internet. Only expose the Laravel web ports (usually
80/443) through the Azure firewall and VM firewall.

To update the ML service later, pull the code as its service account, reinstall
dependencies, and restart the API:

```bash
sudo -u pgpc-ml git -C /opt/machine_learning pull
sudo bash /opt/machine_learning/deploy/install-ubuntu.sh
```

The installer repeats the database snapshot and model training during updates.
If you only want to deploy code without refreshing/retraining, install updated
requirements in `.venv` and run `sudo systemctl restart pgpc-ml-api` instead.

## API security

Every `/api/*` endpoint requires the configured `API_KEY`. Requests without
it receive `401`; repeated requests are limited by `RATE_LIMIT_REQUESTS` during
`RATE_LIMIT_WINDOW_SECONDS` and receive `429` with a `Retry-After` header.
`MAX_REQUEST_BYTES` limits JSON request size. The systemd service runs Gunicorn
with a 30-second worker timeout by default.

## PostgreSQL snapshot and recommender

The model reads interactions directly from PostgreSQL `user_histories`, joined
to accounts for their random 256-bit `ml_user_token`. Numeric account IDs and
legacy `encrypter_id` values are not given to SVD. Ratings are `1` (view) and
`2` (borrowed). Generated seeder rows are marked `is_seeded` and excluded from
training by default; set `ML_INCLUDE_SEEDED_DATA=true` only for a local/demo
model. No interaction CSV is used. Real checkout transactions now update the
student's borrowed rating, while inside-library use is excluded.

The same refresh streams catalog metadata from the library database in batches
of 128 and writes 384-dimensional embeddings to `ml_book_embeddings`. It
embeds titles, subtitles, descriptions, notes, Library of Congress
classification, call numbers, categories, subjects, and authors. It hashes
each record and only re-embeds records whose metadata changed, then removes
records no longer in the catalog. The HNSW index retrieves nearby titles for
personal content matching.

For a large catalog, set `ML_INDEX_DB_DATABASE` and its related `ML_INDEX_DB_*`
settings to a dedicated PostgreSQL database to isolate search storage and
indexing load from the library database. If omitted, the index uses the `DB_*`
database. The index database must have the pgvector extension installed and
available to the ML database user. The metadata sent to the encoder is still
catalog metadata (title, description, subjects, categories, and authors); this
does not ingest or index the full contents of book files.

The refresh also builds `ml_book_year_stats` and `ml_book_popularity` in the
ML index database. They use distinct students with checkout transactions in
the past year; inside-library use is excluded. Year/program statistics require
at least three student borrowers and compare a smoothed group borrowing share
with the share among other groups. Confidence is `borrowers / (borrowers + 10)`.
Only transactions with program/year snapshots are used for group statistics;
new checkouts save those snapshots so later changes to a student's profile do
not relabel old loans. The separate `ml_book_popularity` table supports a
small novelty quota from less-borrowed titles.

`POST /api/search` uses `sentence-transformers/all-MiniLM-L6-v2` embeddings and
cosine similarity to return semantic matches. Example body:

```json
{"query":"introductory books about designing databases", "top_n":10}
```

The Laravel OPAC uses these ranked ISBNs for relevance searches and keeps its
existing database search as a fallback when the ML API is unavailable. The
installer downloads the embedding model into `/var/cache/pgpc-ml/huggingface`.

Run these commands from this directory after the PostgreSQL settings are in
`.env`. Apply the Laravel migrations first so checkout snapshots and the
`is_seeded` history flag exist:

```powershell
python snapshot_user_histories.py
python train_svd_recommender.py
python evaluate_recommender.py
```

To refresh from PostgreSQL and train in one step:

```powershell
python train_svd_recommender.py --refresh-database
```

The training command creates `library_svd_model.pkl` and `all_books.json`.
The API reads those files when `POST /api/recommend` is called:

```json
{
  "user_key": "random-account-token",
  "borrowed_books": ["9780000000000"],
  "student_context": {"program":"BSIT", "year_level":"1st Year"}
}
```

Recommendation ranking combines SVD (50%), similarity to the student's
interaction metadata profile (30%), and year/program borrowing relevance
(20%); missing signals are renormalized. Borrowed books weigh twice as much as
views in the content profile. Up to 30% of the results may be less-borrowed
titles, provided their relevance is close to the best candidates. Current
available-copy count is a tie-breaker and is returned with each result. Laravel
loads the live count again for display. The response keeps the existing ISBN
and `predicted_score` fields and adds `available_copies`.

For the identity migration, first run Laravel's `php artisan migrate` so each
account gets its random token. Then rebuild the model with
`python train_svd_recommender.py --refresh-database` before routing requests to
the updated Laravel and ML services. This reads current histories under the
new token and keeps each active student's SVD history together.
Historical loans created before the checkout snapshot migration have no
program/year-at-issue values, so they don't contribute to year-level scores.

`evaluate_recommender.py` writes the latest SVD holdout metrics and per-signal
comparisons to `model_metrics.json`
and appends every evaluation to `model_metrics_history.jsonl`, so rerunning
the model does not delete previous evaluation results. The API exposes the
latest result at `/api/metrics` and the complete newest-first history at
`/api/metrics/history`. Each history record includes RMSE and MAE,
the total number of interactions and unique users used to train the model,
sample counts, mean/median/min/max/standard deviation for actual ratings,
predicted ratings, and absolute errors, NDCG@10 ranking statistics, and a
generated conclusion. `signal_evaluation` reports RMSE, MAE, and NDCG@10 for
SVD, borrowed-book content similarity, program/year relevance, and the serving
hybrid score on the same held-out interactions. It also reports coverage for
content/year scores and popularity/availability lookups. Popularity and
availability are ranking adjustments in serving, not separate trained models,
so they are coverage diagnostics rather than rating predictors. These metrics
are diagnostic only: the random holdout contains observed interactions but no
impressions or sampled unborrowed books, so they do not measure online top-N
recommendation quality. Year/program statistics are read from the current
index and can include interactions that also appear in the holdout.
NDCG@10 is calculated per user from held-out interactions, using the true
interaction rating as relevance and the signal score as ranking score.

Run the snapshot and training commands from this directory with the venv:

```bash
.venv/bin/python snapshot_user_histories.py
.venv/bin/python train_svd_recommender.py
```
