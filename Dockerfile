# Build on a connected host, push to the internal registry, run air-gapped. NOT yet built or tested
# (no Docker daemon was available during development): build it and run tools/load_test.py before relying on it.
# Pin the base image by digest for a build that can be repeated exactly:
#   docker pull python:3.12-slim && docker inspect --format '{{index .RepoDigests 0}}' python:3.12-slim
#   docker build --build-arg BASE_IMAGE=python@sha256:<digest> ...
ARG BASE_IMAGE=python:3.12-slim
FROM ${BASE_IMAGE}

RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-ara tesseract-ocr-eng openssl libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Two sizes. The default "full" image has PaddleOCR (needed for fine-tuned models). The "lite" image leaves it out and is
# several times smaller:   docker build --build-arg VARIANT=lite -t visa-ocr:lite .
# PaddlePaddle 3.3.1 has no wheel for Linux on ARM. On an Apple-chip Mac build the full image for the servers' architecture:
#   docker build --platform linux/amd64 -t visa-ocr:latest .
# or build the lite image natively (--build-arg VARIANT=lite).
ARG VARIANT=full
WORKDIR /srv
# LOCKED=1 (default) installs from the .lock file: every package, direct and indirect, must match its recorded SHA-256, so a
# changed or swapped package stops the build. LOCKED=0 installs from the plain pinned list (use it only to regenerate a lock).
ARG LOCKED=1
COPY requirements.txt requirements-lite.txt requirements.lock requirements-lite.lock ./
RUN if [ "$VARIANT" = "lite" ]; then REQ=requirements-lite; else REQ=requirements; fi \
    && if [ "$LOCKED" = "1" ]; then pip install --no-cache-dir --no-compile --require-hashes -r "$REQ.lock"; \
       else pip install --no-cache-dir --no-compile -r "$REQ.txt"; fi \
    && rm -f requirements*.txt requirements*.lock

COPY app ./app
COPY config ./config
COPY VERSION ./VERSION
COPY docker-entrypoint.sh ./docker-entrypoint.sh
# Models are fetched on the build host (python tools/fetch_models.py) and copied in; never downloaded at run time.
COPY models ./models

# "auto" uses PaddleOCR when it is installed and its models are present, and the Tesseract + RapidOCR pair otherwise.
# This is a production image: it starts in production mode and refuses to start in any unsafe set-up (no key file, no
# certificate, stored data not encrypted). PRODUCTION=0 gives the test mode, for a workstation only.
ENV PRODUCTION=1 \
    OCR_ENGINE=auto \
    MODELS_DIR=/srv/models \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    PYTHONUNBUFFERED=1 \
    HOME=/data/home \
    PADDLE_PDX_CACHE_HOME=/data/home/.paddlex \
    JOB_DB_PATH=/data/jobs.db

# The copied files keep the permissions they had on the build host. A checkout with owner-only files (umask 077) would leave
# the service user unable to read its own code, so make everything under /srv readable here, whatever the host had.
RUN useradd --system --no-create-home ocr && mkdir /data && chown ocr /data \
    && chmod -R a+rX /srv && chmod 755 docker-entrypoint.sh
USER ocr
VOLUME /data
# Binding port 443 as a non-root user needs this capability on the container (Docker grants NET_BIND_SERVICE by default).
EXPOSE 443
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s CMD python -c "import ssl,urllib.request as u;u.urlopen('https://127.0.0.1:443/readyz',context=ssl._create_unverified_context())" || exit 1
CMD ["./docker-entrypoint.sh"]
