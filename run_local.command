#!/bin/zsh
# Double-click to start the document reader on this Mac. Close the window (or press Ctrl+C) to stop it.
cd "$(dirname "$0")"
export OCR_ENGINE=ensemble OCR_DEADLINE_SECONDS=1.7 RATE_LIMIT_PER_MIN=0
export JOB_DB_PATH="$PWD/.local/jobs.db" TRAINING_STORE_DIR="$PWD/.local/training"
mkdir -p .local
lsof -ti tcp:8080 | xargs kill 2>/dev/null
(sleep 4; open "http://127.0.0.1:8080/ui") &
exec .venv/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8080
