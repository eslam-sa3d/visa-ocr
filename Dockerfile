# Build on a connected host, push to the internal registry, run air-gapped. NOT yet built or tested
# (no Docker daemon was available during development): build it and run tools/load_test.py before relying on it.
FROM python:3.12-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends tesseract-ocr tesseract-ocr-ara tesseract-ocr-eng openssl libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY config ./config
COPY docker-entrypoint.sh ./docker-entrypoint.sh
# Models are fetched on the build host (python tools/fetch_models.py) and copied in; never downloaded at run time.
COPY models ./models

ENV OCR_ENGINE=auto \
    MODELS_DIR=/srv/models \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    PYTHONUNBUFFERED=1 \
    JOB_DB_PATH=/data/jobs.db

RUN useradd --system --no-create-home ocr && mkdir /data && chown ocr /data && chmod +x docker-entrypoint.sh
USER ocr
VOLUME /data
# Binding port 443 as a non-root user needs this capability on the container (Docker grants NET_BIND_SERVICE by default).
EXPOSE 443
HEALTHCHECK --interval=30s --timeout=5s --start-period=90s CMD python -c "import ssl,urllib.request as u;u.urlopen('https://127.0.0.1:443/readyz',context=ssl._create_unverified_context())" || exit 1
CMD ["./docker-entrypoint.sh"]
