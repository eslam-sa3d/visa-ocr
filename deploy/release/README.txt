visa-ocr @VERSION@ (build @TAG@) - production container image for Intel/AMD Linux servers (linux/amd64)
==========================================================================================

Files
  visa-ocr-@TAG@-amd64.tar.gz   the image
  production/                   settings file, compose file and key maker for the production set-up
  DEPLOYMENT.txt                step-by-step notes for the server team (start here)
  GO_LIVE_RECORD.md             what was proven for this build and what is still open before go-live
  docs.tar.gz, tools.tar.gz     documentation; test and measurement tools with the source they need
  BUILD_INFO.txt                base image digest, image id, build date, test results
  SHA256SUMS.txt                checksums of everything above

The image needs NO internet at any point: models and all packages are inside it.

This is a PRODUCTION image. It starts in production mode by default and refuses to start, saying why, unless it has API
keys from a key file, an encryption key for stored data and your organisation's TLS certificate.

1. Check the files arrived intact
     sha256sum -c SHA256SUMS.txt            (on a Mac: shasum -a 256 -c SHA256SUMS.txt)

2. Load the image
     gunzip -c visa-ocr-@TAG@-amd64.tar.gz | docker load

3. Run it in production: follow DEPLOYMENT.txt (about ten minutes).

4. Only to try it on a workstation (test mode: self-signed certificate, one key, test page at /ui):
     docker run -d --name visa-ocr-test -p 8443:443 -e PRODUCTION=0 -e API_KEYS="$(openssl rand -hex 32)" visa-ocr:@TAG@
     curl -k https://localhost:8443/readyz
   Never use test mode with real documents.

What is inside
  Readers: PaddleOCR (English, Arabic), Tesseract with the project's own OCR-B model for MRZ print, RapidOCR.
  Endpoints: POST /v1/documents/extract, GET /v1/jobs/{id}, POST /v1/documents/check, POST /v1/mrz/parse,
             POST /v1/jobs/{id}/corrections, GET /v1/models.

Known limits
  Built for linux/amd64. On an ARM machine it runs emulated and slowly; that is for trying it only.
  Passports, visas and cards are read from the side that carries the MRZ, where check digits confirm the reading.
  Saudi, UAE, Kuwait, Qatar, Bahrain and Oman cards, and the FRONT of European, UK and US residence cards, are read from
  their printed text. Those sides have no check digit, so every such reading goes to a person for review.
  Open before go-live: see GO_LIVE_RECORD.md.
