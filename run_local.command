#!/bin/zsh
# Double-click to start the document reader on this Mac, with the same readers as the production image (PaddleOCR first,
# then Tesseract + RapidOCR). Test mode: no key, no HTTPS, test page on. Close the window (or press Ctrl+C) to stop it.
# Never use this mode on a server or with documents that must stay private beyond this Mac.
cd "$(dirname "$0")"
export PRODUCTION=0 OCR_ENGINE=auto RATE_LIMIT_PER_MIN=0
export JOB_DB_PATH="$PWD/.local/jobs.db" TRAINING_STORE_DIR="$PWD/.local/training"
mkdir -p .local
lsof -ti tcp:8080 | xargs kill 2>/dev/null
(sleep 12; open "http://127.0.0.1:8080/ui") &
exec .venv-paddle/bin/python -m uvicorn app.main:app --host 127.0.0.1 --port 8080
