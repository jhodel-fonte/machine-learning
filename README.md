# machine-learning

## Azure Docker deployment

This service can be deployed as a Docker container to Azure App Service for
Containers, Azure Container Apps, or Azure Container Instances. Build and push
the image to Azure Container Registry:

```powershell
az acr build --registry <registry-name> --image recommender-api:latest .
```

Deploy the image to an Azure container service and configure these environment
variables in the Azure portal or deployment command:

```text
environment
API_KEY
DB_HOST
DB_PORT
DB_DATABASE
DB_USERNAME
DB_PASSWORD
RATE_LIMIT_REQUESTS
RATE_LIMIT_WINDOW_SECONDS
MAX_REQUEST_BYTES
REQUEST_TIMEOUT_SECONDS
AUTO_TRAIN_ENABLED
AUTO_TRAIN_TIME
AUTO_TRAIN_TIMEZONE
```

Do not upload `.env` to the image or commit it to Git. Store credentials in
Azure Key Vault or the service's managed secret settings.

Use `environment=local` with `DB_HOST=localhost` when running the Python
service directly on your development machine. For local Docker, set
`DB_HOST_CONTAINER=host.docker.internal`. For Docker on the Azure VM, set
`environment=production` (or `azure`) and set `DB_HOST_CONTAINER` to the
PostgreSQL container's Compose service name, such as `postgres`. The Compose
file uses `DB_HOST_CONTAINER` only inside containers; `DB_HOST` remains
available for direct Python execution.

Generated files such as `library_svd_model.pkl`, `testset.pkl`,
`all_books.json`, and `model_metrics.json` are intentionally excluded from
Git and the Docker build context. Train the model before starting the API, or
store/download these artifacts from durable storage such as Azure Blob
Storage during deployment. The API container is stateless and its bind mount
in `docker-compose.yml` is intended for local development only.

### Ubuntu Azure VM with Docker Compose

For an Ubuntu VM, the API and trainer are separate containers intentionally:

- `recommender-api` serves requests and loads the trained model.
- `recommender-trainer` refreshes data, trains the model, evaluates it, and
  then sleeps until the next interval.
- A slow training job therefore does not block recommendation requests.

They can still access the same model. In the current Compose configuration,
both services mount the same VM directory at `/app`. The trainer writes
`library_svd_model.pkl`, `all_books.json`, and `model_metrics.json` into that
directory, and the API reads those files from the same directory. Training
should be allowed to finish before the API starts. For zero-downtime
production updates, publish versioned model artifacts to durable storage and
switch the API to a completed version instead of replacing files in place.

On the VM:

```bash
git clone <repository-url> machine_learning
cd machine_learning
cp .env.example .env
nano .env
docker compose build
docker compose run --rm recommender-trainer python snapshot_user_histories.py
docker compose run --rm recommender-trainer python train_svd_recommender.py
docker compose run --rm recommender-trainer python evaluate_recommender.py
docker compose up -d recommender-api
```

For scheduled training at midnight, set `AUTO_TRAIN_ENABLED=true`,
`AUTO_TRAIN_TIME=00:00`, and the VM timezone in `.env`, then run:

```bash
docker compose --profile autotrain up -d recommender-api recommender-trainer
```

For Philippine time, use:

```env
AUTO_TRAIN_ENABLED=true
AUTO_TRAIN_TIME=00:00
AUTO_TRAIN_TIMEZONE=Asia/Manila
```

The trainer waits for the next configured midnight before training; it does
not train immediately when the container starts. Use an IANA timezone such as
`UTC`, `Asia/Manila`, or `America/New_York`.

### Separate PostgreSQL container on the Azure VM

Because PostgreSQL runs in another container, both containers must be attached
to the same Docker network. Set `DB_HOST` to the PostgreSQL Compose service
name, not `localhost`. For example, if the database service is named
`postgres`, use:

```env
DB_HOST=postgres
DB_HOST_CONTAINER=postgres
DB_PORT=5432
```

The database container does not need to publish port `5432` to the public
internet. Containers communicate over the internal Docker network. If the
database and ML service are defined in separate Compose projects, create a
shared external network and attach both projects to it:

```bash
docker network create pgpc-internal
```

Then configure both Compose files with an external network named
`pgpc-internal`. The ML service's `DB_HOST` must match the database service
name or a network alias on that shared network. Verify connectivity from the
ML container with:

```bash
docker compose run --rm recommender-api python -c \
  "import os, psycopg2; c=psycopg2.connect(host=os.environ['DB_HOST'], \
  port=os.environ['DB_PORT'], dbname=os.environ['DB_DATABASE'], \
  user=os.environ['DB_USERNAME'], password=os.environ['DB_PASSWORD']); \
  print('database connection ok'); c.close()"
```

For this architecture, use one of these values:

- Azure Database for PostgreSQL: set `DB_HOST` to the Azure PostgreSQL
  hostname.
- PostgreSQL in another container on the same Docker network: use that
  container's Compose service name, such as `postgres`.
- PostgreSQL running directly on the VM: use the VM's Docker gateway address
  or a configured host alias.

Do not use `DB_HOST=localhost` inside the ML container. In a container,
`localhost` means the ML container itself. Open only the required VM ports
(normally `80`/`443` through a reverse proxy); do not expose PostgreSQL port
`5432` publicly.

## API security and automatic training

Copy `.env.example` to `.env` and set a long random `API_KEY`. Every
`/api/*` endpoint requires that key:

```powershell
curl -H "X-API-Key: your-secret-key" http://localhost:5000/api/metrics
```

Requests without the key receive `401`. Repeated requests are limited by
`RATE_LIMIT_REQUESTS` during `RATE_LIMIT_WINDOW_SECONDS` per API key and
receive `429` with a `Retry-After` header when the limit is exceeded.
`MAX_REQUEST_BYTES` limits JSON request size. Gunicorn stops a request after
`REQUEST_TIMEOUT_SECONDS` (30 seconds by default).

Automatic training is disabled by default. To enable the separate trainer,
set `AUTO_TRAIN_ENABLED=true` and choose `AUTO_TRAIN_TIME` and
`AUTO_TRAIN_TIMEZONE`, then
start the Compose profile:

```powershell
docker compose --profile autotrain up -d recommender-trainer
```

The trainer refreshes `user_histories`, trains the SVD model, and waits for
the configured interval. Keep it as a separate service so a slow training run
cannot block API requests. Do not enable it until the database contains
production-ready data.

## PostgreSQL snapshot and recommender

The model uses `dataset/interactions.csv` with these columns:

```text
User-ID,ISBN,rating
```

The machine-learning service reads only `encrypter_id`, `isbn`, and `rating`
from `user_histories`. It never selects or exports `user_id`. Rows without an
`encrypter_id` are excluded. Ratings are interpreted as `0` (no interaction),
`1` (view), and `2` (borrowed).

The snapshot refuses to replace an existing dataset when the table is empty.
This protects the training data if the wrong database or an unseeded database
is configured. Use `--allow-empty` only when an empty dataset is intentional.

Run these commands from this directory after the PostgreSQL settings are in
`.env`:

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
  "encrypter_id": "encrypted-user-id",
  "borrowed_books": ["9780000000000"]
}
```

`evaluate_recommender.py` writes `model_metrics.json` with RMSE and MAE,
sample counts, mean/median/min/max/standard deviation for actual ratings,
predicted ratings, and absolute errors, NDCG@10 ranking statistics, and a
generated conclusion. NDCG@10 is calculated
per user from the held-out test interactions, using the true interaction
rating as relevance and the predicted score as the ranking score.

For Docker, run the snapshot and training commands inside the service:

```powershell
docker compose run --rm recommender-api python snapshot_user_histories.py
docker compose run --rm recommender-api python train_svd_recommender.py
```

`docker-compose.yml` maps the host PostgreSQL server as
`host.docker.internal`; do not change the container database host to
`localhost`.
