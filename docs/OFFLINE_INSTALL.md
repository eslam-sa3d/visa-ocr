# Offline install (air-gapped host)

On a connected machine with the **same OS/CPU and Python 3.12**:
```
pip download -r requirements.txt -d wheelhouse          # about 84 wheels, ~400 MB
python tools/fetch_models.py                            # fills models/ with sha256-checked OCR models
```
Copy `wheelhouse/`, `models/` and the project to the offline host, then:
```
python3.12 -m venv .venv && . .venv/bin/activate
pip install --no-index --find-links wheelhouse -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8080
```
Checked: every pin in `requirements.txt` resolves to a wheel for CPython 3.12 / Linux x86-64.
Docker: build on a connected host (`docker build -t visa-ocr .`), push to the internal registry. The Dockerfile has **not** been built by the author (the base image could not be pulled in the build environment); run `docker build` and `tools/load_test.py` against the container before go-live.

## Offline guarantees (verified)
- The application code contains no outbound network calls (no downloads, telemetry, CDN assets; the test page loads nothing external).
- PaddleOCR is forced offline in code (`PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK`, `HF_HUB_OFFLINE`); models load only from `models/`
  and are refused if missing or if the sha256 differs.
- Tested with the network namespace disabled (no route to anything outside the machine): the server started over HTTPS,
  `/readyz` was ready, and a passport image was uploaded and read correctly.
- Internet is needed only for the one-time build steps: `pip download`, `tools/fetch_models.py`, `docker build`.
- Check it yourself in Docker: `docker network create --internal nonet` then
  `docker run --rm --network nonet -e API_KEYS=... visa-ocr:0.3` and call it from another container on that network.
