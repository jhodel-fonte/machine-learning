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
for Python packages, the embedding model cache, and database index growth. The
trained model stays under `/opt/machine_learning`; held-out interactions and
catalog data are stored in PostgreSQL. The directory also contains the Python
environment, configuration, embedding model cache, and evaluation reports.

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

To update the ML service later, pull the code as its service account, then
run the update script:

```bash
sudo -u pgpc-ml git -C /opt/machine_learning pull
sudo bash /opt/machine_learning/deploy/update-ubuntu.sh
```

The updater reinstalls Python requirements and service definitions, then restarts
both services. It preserves `.env` and does not refresh data or retrain the model.

## API security

Every `/api/*` endpoint requires the configured `API_KEY`. Requests without
it receive `401`; repeated requests are limited by `RATE_LIMIT_REQUESTS` during
`RATE_LIMIT_WINDOW_SECONDS` and receive `429` with a `Retry-After` header.
`MAX_REQUEST_BYTES` limits JSON request size. The systemd service runs Gunicorn
with a 30-second worker timeout by default.

## PostgreSQL snapshot and recommender

For an interactive training, refresh, evaluation, and API menu, run from this
directory on Windows or Ubuntu:

```bash
python ml_menu.py
```

On Ubuntu, use `python3 ml_menu.py` if `python` is unavailable. Choose Docker
Compose when running the containers, or native Python for the systemd deployment.
The native option uses `.venv` if present and runs commands as the current user;
that user needs access to `.env`, the model files, and the database. On the
systemd deployment, run the menu as the `pgpc-ml` service account when needed.
The demo-data choice applies to commands launched by the menu only. Set
`ML_INCLUDE_SEEDED_DATA=true` in `.env` as well if the API should use seeded
histories. Catalog refresh still requires pgvector in the index database.
Menu option 10 evaluates the existing model and held-out interactions against
active ISBNs read directly from the library database, then saves the results for
the dashboard. It does not refresh embeddings, retrain the model, or change the
held-out data; those data may include seeded interactions. Refresh the dashboard
afterward. The content and hybrid evaluations still require the separate index.
Menu option 7 deletes a selected model, metrics, held-out interactions, index,
seeded histories, or all user histories. Database actions display the target
database and require typing `DELETE`; deleting all histories includes real
student interactions. Rebuild the index with refresh and the model with
training after deleting them. Restart the API after replacing a model.
In native mode, option 8 runs the Ubuntu installer from `/opt/machine_learning`.
Configure `.env` first, including `ML_INCLUDE_SEEDED_DATA` if using demo data;
the installer's database refresh and training use that file rather than the
menu's temporary demo-data choice. PostgreSQL and pgvector must already be set up.
After `git pull`, option 9 updates the native services or rebuilds the Docker
Compose services. Both options require an existing `.env` and preserve it.
Option 9 does not run `git pull`, refresh the index, or retrain the model.
In Docker Compose mode, option 11 builds the images, refreshes the catalog,
trains and evaluates the model, and starts both ML services in one action. It
uses the menu's demo-data choice for those commands. Install and start Docker
with Compose first, configure `.env`, and ensure PostgreSQL, pgvector, and the
external `pgpc-internal` Docker network are ready. The refresh and training
update database data and generated model and evaluation files.

The model reads interactions directly from PostgreSQL `user_histories`, joined
to accounts for their random 256-bit `ml_user_token`. Numeric account IDs and
legacy `encrypter_id` values are not given to SVD. Ratings are `1` (view) and
`2` (borrowed). Generated seeder rows are marked `is_seeded` and excluded from
training by default; set `ML_INCLUDE_SEEDED_DATA=true` only for a local/demo
model. No interaction CSV or local interaction snapshot is used. Real checkout transactions now update the
student's borrowed rating, while inside-library use is excluded.

For a demo, the Laravel application can import a CSV with `date` (or `date_borrowed`) and
`accession_number` columns:

```powershell
cd ../pgpc-lib
php artisan demo:import-borrowings path/to/loans.csv
```

The command matches accessions to catalog ISBNs and assigns loans to existing
student accounts in small groups based on call-number or classification
prefixes. These assignments are synthetic, not actual borrower histories. It
matches numeric accessions despite leading zeroes when the match is unique. It
marks new interactions as seeded, skips unmatched or invalid rows, and leaves
existing interactions unchanged. Repeating an import does not overwrite them.
Set `ML_INCLUDE_SEEDED_DATA=true` in the ML `.env` before training a demo model;
remove that setting and retrain for a model based on real interactions only.

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

The training command saves the trained SVD model as `library_svd_model.pkl`
and stores the held-out evaluation interactions in the PostgreSQL table
`ml_recommender_test_interactions`. The catalog is read directly from
`ml_book_embeddings`; it no longer writes or reads `testset.pkl` or
`all_books.json` data snapshots. Database backups include the source histories,
catalog index, borrowing statistics, and held-out evaluation rows.
Keep `library_svd_model.pkl`; the API and evaluator still load this trained
model artifact from disk.

`POST /api/recommend` accepts:

```json
{
  "user_key": "random-account-token",
  "borrowed_books": ["9780000000000"],
  "student_context": {"program":"BSIT", "year_level":"1st Year"},
  "interaction": {"isbn":"9780000000001", "rating":1}
}
```

Recommendation ranking combines SVD (20%), similarity to the student's
interaction metadata profile (70%), and year/program borrowing relevance
(10%); missing signals are renormalized. The content profile uses current
history on each request, including author names in indexed book metadata. When
an interaction ISBN is supplied, its indexed metadata drives content similarity
immediately, even for a first view; if unavailable, the history profile is used.
The opened book is excluded from that request's recommendations.
Borrowed books weigh twice as much as
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
comparisons to `model_metrics.json` and appends every evaluation to
`model_metrics_history.jsonl`, so rerunning the model does not delete previous
evaluation results. The API exposes the
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

The same report also includes `recommendation_evaluation.strategies`, which
ranks the full indexed catalog separately for SVD-only, content-only,
program/year-only, popularity-only, hybrid-score, hybrid-with-availability,
and hybrid rankings with availability and novelty. Each strategy reports macro Precision@10,
Recall@10, NDCG@10, catalog coverage, availability rate, and less-popular-item
rate. These rankings exclude books seen in that user's training history;
popularity and cohort scores are built from training interactions only. The
evaluation still uses the existing random 80/20 split and current student
program/year, treats held-out views/borrows as relevant, and has no exposure
logs, so unborrowed candidates are not confirmed negatives. Availability is
the current copy count, not historical availability at test time.

Run the snapshot and training commands from this directory with the venv:

```bash
.venv/bin/python snapshot_user_histories.py
.venv/bin/python train_svd_recommender.py
```
