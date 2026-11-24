#!/bin/zsh
# Double-click to read every image in ~/Development/visa-ocr-testdata with the production readers and write the results to a
# new results folder beside it. The folder holds what was read from the documents (sheet.csv): it is created for your account
# only; delete it when the checking is done. Takes about an hour; keep the Mac awake. Stop the local service first.
cd "$(dirname "$0")"
export PRODUCTION=0 OCR_ENGINE=auto
n=8; while [ -e "$HOME/Development/visa-ocr-testdata-results$n" ]; do n=$((n+1)); done
out="$HOME/Development/visa-ocr-testdata-results$n"
lsof -ti tcp:8080 | xargs kill 2>/dev/null
caffeinate -i .venv-paddle/bin/python tools/read_folder.py "$HOME/Development/visa-ocr-testdata" --out "$out"
echo; echo "Finished. Results are in $out"; cat "$out/summary.txt"
