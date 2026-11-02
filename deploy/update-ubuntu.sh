#!/usr/bin/env bash
set -euo pipefail

APP_DIR=/opt/machine_learning
if [[ $EUID -ne 0 ]]; then
  echo "Run this updater with sudo." >&2
  exit 1
fi
cd "$APP_DIR"
if [[ ! -f .env || ! -x .venv/bin/python ]]; then
  echo "Missing .env or Python environment; configure .env and run the installer first." >&2
  exit 1
fi

runuser -u pgpc-ml -- "$APP_DIR/.venv/bin/python" -m pip install --no-build-isolation -r requirements.txt
install -m 644 deploy/pgpc-ml-api.service /etc/systemd/system/pgpc-ml-api.service
install -m 644 deploy/pgpc-ml-trainer.service /etc/systemd/system/pgpc-ml-trainer.service
systemctl daemon-reload
systemctl restart pgpc-ml-api.service pgpc-ml-trainer.service
echo "ML services updated and restarted. Existing .env and model were preserved."
