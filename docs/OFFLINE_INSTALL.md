# Offline install (air-gapped host)

On a connected machine with the **same OS/CPU and Python 3.12**:
```
pip download -r requirements.txt -d wheelhouse          # about 84 wheels, ~400 MB
python tools/fetch_models.py                            # fills models/ with sha256-checked OCR models
```
Copy `wheelhouse/`, `models/` and the project to the offline host, then:
```
python3.12 -m venv .venv && . .venv/bin/activate
pip install --no-index --find-links wheelhouse --require-hashes -r requirements.lock
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

## Verifying what you carry across (added after the engineering and ML review)

On the connected machine, after `pip download -r requirements.lock -d wheelhouse` and `python tools/fetch_models.py`:

```
python tools/make_checksums.py wheelhouse          # writes wheelhouse/SHA256SUMS
```

On the offline server, before installing:

```
python tools/make_checksums.py --check wheelhouse  # every file must match; nothing extra may be present
pip install --no-index --find-links wheelhouse --require-hashes -r requirements.lock
```

`requirements.lock` and `requirements-lite.lock` list every package, direct and indirect, with the SHA-256 of its files, so
pip itself refuses a changed wheel. They were generated on Python 3.13; regenerate them on the Python version of the target
(`pip-compile --generate-hashes --allow-unsafe --strip-extras -o requirements.lock requirements.txt`) if the install reports
a missing package. The image build uses the lock by default (`LOCKED=1`) and can pin the base image by digest
(`--build-arg BASE_IMAGE=python@sha256:...`).

Prove the image works with no network before it leaves the build host:

```
tools/offline_smoke.sh visa-ocr:latest path/to/a_document.jpg
```

It starts the container with `--network none`, uploads the document from inside it and expects a finished read. This is the
only check that exercises PaddleOCR and PaddleX with the network truly absent; the test suite blocks sockets inside Python
only (`tests/test_review_fixes.py`).

Without Docker, `sudo tools/offline_selftest.sh` gives the same proof for an installed copy: it starts the service inside a
network namespace that has only the loopback interface, checks that the outside world is unreachable from there, and runs
the functional system test (19 checks). It was run this way for the Tesseract + RapidOCR engine and passed. Run it on the
real server with `OCR_ENGINE` as it will be in production, so that PaddleOCR is covered too.
