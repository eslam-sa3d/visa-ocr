# visa-ocr: shareable folder

Offline FastAPI OCR service for passports, Schengen visas, ID cards, Iqama and GCC IDs (English and Arabic).

1. Read `README.md` first, then `docs/ACCEPTANCE_SIGNOFF.md` (decision sheet for the business owner).
2. Run it: `docker build -t visa-ocr .` then `docker run -p 8000:8000 visa-ocr`, or see README for a local venv.
3. Optional Paddle models: follow `docs/RUNBOOK_CONNECTED_MACHINE.md` on a machine with internet access.
4. Tests: `pip install -r requirements-dev.txt && pytest -q` (178 pass, 6 skipped).

Status: supervised pilot only. Accuracy figures in the docs are on generated documents, not real ones. No real personal documents are included in this folder.
