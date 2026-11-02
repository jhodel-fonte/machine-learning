"""Interactive commands for Docker Compose and native ML installations."""

import os
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parent


def run(command, env=None):
    result = subprocess.run(command, cwd=ROOT, env=env, check=False)
    if result.returncode:
        print(f"Command failed with exit code {result.returncode}.")
    return result.returncode == 0


def main():
    mode = input("Run using [1] Docker Compose or [2] native Python? ").strip()
    if mode not in {"1", "2"}:
        print("Choose 1 or 2.")
        return

    seeded = input("Include seeded demo interactions for training and evaluation? [y/N] ").strip().lower() in {"y", "yes"}
    env = os.environ.copy()
    if seeded:
        env["ML_INCLUDE_SEEDED_DATA"] = "true"
    venv_python = ROOT / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    python = str(venv_python) if venv_python.exists() else sys.executable

    def script(name, *args):
        if mode == "1":
            command = ["docker", "compose", "run", "--rm"]
            if seeded:
                command += ["-e", "ML_INCLUDE_SEEDED_DATA=true"]
            return run(command + ["recommender-trainer", "python", name, *args], env)
        return run([python, name, *args], env)

    while True:
        print("\n=== ML Menu ====================================================")
        print("\n1. Train model\n2. Evaluate model\n3. Refresh catalog and statistics\n"
              "4. Refresh, train and evaluate\n5. Start/restart API\n6. Check API health\n"
              "7. Delete generated data or interactions\n"
              "8. Install native Ubuntu deployment\n9. Apply updates after git pull\n"
              "10. Evaluate using current database catalog\n0. Exit")
        choice = input("Choose an action: ").strip()
        if choice == "0":
            return
        if choice == "1":
            script("train_svd_recommender.py")
        elif choice == "2":
            script("evaluate_recommender.py")
        elif choice == "3":
            script("snapshot_user_histories.py")
        elif choice == "4":
            if script("snapshot_user_histories.py") and script("train_svd_recommender.py"):
                script("evaluate_recommender.py")
        elif choice == "5":
            command = (["docker", "compose", "up", "-d", "--force-recreate", "recommender-api"]
                       if mode == "1" else ["sudo", "systemctl", "restart", "pgpc-ml-api"])
            run(command, env)
        elif choice == "6":
            command = (["docker", "compose", "exec", "-T", "recommender-api", "python", "-c",
                        "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:5000/health', timeout=5).read().decode())"]
                       if mode == "1" else [python, "-c", "import urllib.request; print(urllib.request.urlopen('http://127.0.0.1:8005/health', timeout=5).read().decode())"])
            run(command, env)
        elif choice == "10":
            print('Uses the current catalog and the existing model and held-out data; it does not retrain the model.')
            if script('evaluate_recommender.py', '--catalog-from-database'):
                print('Evaluation saved. Refresh results in the system dashboard.')
        elif choice == "7":
            print("\n1. Trained model file\n2. Evaluation metrics and history files\n"
                  "3. Held-out evaluation interactions\n4. Catalog embeddings and statistics\n"
                  "5. Seeded demo user histories\n6. ALL user histories (including real interactions)\n0. Back")
            selection = input("Choose what to delete: ").strip()
            actions = {"1": "model", "2": "metrics", "3": "test-interactions",
                       "4": "index", "5": "seeded-histories", "6": "all-histories"}
            if selection in actions:
                script("ml_cleanup.py", actions[selection])
            elif selection != "0":
                print("Choose an option from the menu.")
        elif choice in {"8", "9"}:
            if choice == "8" and mode != "2":
                print('Ubuntu installation requires native mode.')
                continue
            if mode == "2" and (sys.platform != "linux" or ROOT != Path('/opt/machine_learning')):
                print('Native Ubuntu deployment requires a checkout at /opt/machine_learning.')
                continue
            if not (ROOT / '.env').is_file():
                print('Missing .env. Copy .env.example to .env and configure it before continuing.')
                continue
            print('Existing .env found; its contents will be preserved.')
            if choice == "8":
                print('Installation uses sudo, installs packages and services, changes directory ownership and .env permissions, '
                      'refreshes the index, retrains the model, and evaluates it. PostgreSQL and pgvector must be ready.')
                command = ['sudo', 'bash', str(ROOT / 'deploy/install-ubuntu.sh')]
            elif mode == "2":
                print('Update installs Python dependencies and service definitions, then restarts both ML services. '
                      'It does not pull code or retrain the model.')
                command = ['sudo', 'bash', str(ROOT / 'deploy/update-ubuntu.sh')]
            else:
                print('Update rebuilds and restarts both Docker Compose ML services. It does not pull code or retrain the model.')
                command = ['docker', 'compose', 'up', '-d', '--build', 'recommender-api', 'recommender-trainer']
            if input('Continue? [y/N] ').strip().lower() in {'y', 'yes'}:
                run(command, env)
        else:
            print("Choose an option from the menu.")


if __name__ == "__main__":
    try:
        main()
    except (EOFError, KeyboardInterrupt):
        print("\nExiting.")
    except OSError as error:
        print(f"Could not run command: {error}", file=sys.stderr)
        sys.exit(1)
