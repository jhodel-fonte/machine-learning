#!/usr/bin/env bash
set -euo pipefail

APP_DIR=/opt/machine_learning
if [[ $EUID -ne 0 ]]; then
  echo "Run this installer with sudo." >&2
  exit 1
fi
cd "$APP_DIR"

if [[ ! -f .env ]]; then
  echo "Missing $APP_DIR/.env. Copy .env.example to .env, configure it, then rerun." >&2
  exit 1
fi
if grep -Eq '^API_KEY=("generate-a-long-random-secret"|generate-a-long-random-secret)?$' .env; then
  echo "Set a unique API_KEY in $APP_DIR/.env before installation." >&2
  exit 1
fi

apt-get update
DEBIAN_FRONTEND=noninteractive apt-get install -y build-essential libpq-dev python3-venv python3-dev

if ! id pgpc-ml >/dev/null 2>&1; then
  useradd --system --home-dir "$APP_DIR" --shell /usr/sbin/nologin pgpc-ml
fi
mkdir -p /var/cache/pgpc-ml/huggingface
chown -R pgpc-ml:pgpc-ml "$APP_DIR" /var/cache/pgpc-ml
chmod 640 "$APP_DIR/.env"

python3 -m venv "$APP_DIR/.venv"
"$APP_DIR/.venv/bin/python" -m pip install --upgrade pip setuptools wheel
"$APP_DIR/.venv/bin/python" -m pip install numpy==1.26.4
"$APP_DIR/.venv/bin/python" -m pip install --no-build-isolation -r requirements.txt

runuser -u pgpc-ml -- env HF_HOME=/var/cache/pgpc-ml/huggingface \
  "$APP_DIR/.venv/bin/python" -c \
  "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
runuser -u pgpc-ml -- "$APP_DIR/.venv/bin/python" snapshot_user_histories.py
runuser -u pgpc-ml -- "$APP_DIR/.venv/bin/python" train_svd_recommender.py
runuser -u pgpc-ml -- "$APP_DIR/.venv/bin/python" evaluate_recommender.py

install -m 644 deploy/pgpc-ml-api.service /etc/systemd/system/pgpc-ml-api.service
install -m 644 deploy/pgpc-ml-trainer.service /etc/systemd/system/pgpc-ml-trainer.service
systemctl daemon-reload
systemctl enable --now pgpc-ml-api.service pgpc-ml-trainer.service

echo "ML API is installed at http://127.0.0.1:8005. Check: systemctl status pgpc-ml-api"
