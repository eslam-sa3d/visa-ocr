#!/bin/sh
# Serves HTTPS on PORT (default 443). Use your own certificate: mount it and set TLS_CERT and TLS_KEY.
# Without one, a SELF-SIGNED certificate is generated so the service starts; browsers will warn. Test use only.
set -eu
umask 077        # files the service creates (database, TLS key, training data) are owner-only
PORT="${PORT:-443}"
# The service user has no home folder of its own; libraries that keep a small cache get one on the data volume.
mkdir -p "${HOME:-/data/home}" 2>/dev/null || true
# Same rule as app/security.py: a value made only of commas or spaces is not a key.
KEYS_SET="$(printf '%s%s' "${API_KEYS:-}" "${API_KEY_SHA256:-}" | tr -d ', \t\r\n')"
# Keys with roles and expiry dates come from a file (the production set-up). The service itself checks what is in it.
KEYS_FILE="$(printf '%s' "${API_KEYS_FILE:-}" | tr -d ' \t\r\n')"
if [ -n "$KEYS_FILE" ] && [ ! -s "$KEYS_FILE" ]; then
  echo "ERROR: API_KEYS_FILE is set to $KEYS_FILE, which is missing or empty. Mount the key file there." >&2
  exit 1
fi
if [ -z "$KEYS_SET" ] && [ -z "$KEYS_FILE" ] && [ "${ALLOW_ANONYMOUS:-0}" != "1" ]; then
  echo "ERROR: no usable API key is configured (API_KEYS_FILE, API_KEYS or API_KEY_SHA256). Set one, or set ALLOW_ANONYMOUS=1 for local tests only." >&2
  exit 1
fi
if [ -z "$KEYS_SET" ]; then unset API_KEYS API_KEY_SHA256; fi     # anonymous mode was chosen: drop blank values
CERT="${TLS_CERT:-/data/tls/server.crt}"
KEY="${TLS_KEY:-/data/tls/server.key}"
# Docker creates an empty DIRECTORY when you bind-mount a host path that does not exist (-v /host/server.crt:/data/tls/server.crt).
# A directory passes a size test but cannot be loaded as a certificate, so say what is wrong instead of crashing in uvicorn.
for f in "$CERT" "$KEY"; do
  if [ -d "$f" ]; then
    echo "ERROR: $f is a directory, not a file. The host path you mounted there does not exist (Docker created a folder)." >&2
    echo "       Check the -v path, or mount the folder holding server.crt and server.key (-v /secure/tls:/data/tls:ro)." >&2
    exit 1
  fi
done
if [ ! -f "$CERT" ] || [ ! -s "$CERT" ] || [ ! -f "$KEY" ] || [ ! -s "$KEY" ]; then
  if [ "${PRODUCTION:-0}" = "1" ]; then
    echo "ERROR: PRODUCTION=1 and no TLS certificate at $CERT. Mount your certificate and key (TLS_CERT, TLS_KEY); a self-signed one is not generated in production." >&2
    exit 1
  fi
  echo "WARNING: no TLS certificate at $CERT: generating a self-signed one (test only)" >&2
  mkdir -p "$(dirname "$CERT")"
  openssl req -x509 -newkey rsa:2048 -nodes -days 365 -subj "/CN=${TLS_CN:-localhost}" \
    -addext "subjectAltName=DNS:${TLS_CN:-localhost},DNS:localhost,IP:127.0.0.1" -keyout "$KEY" -out "$CERT" 2>/dev/null
  chmod 600 "$KEY"
fi
# TLS 1.2 and 1.3 only, with forward-secret AEAD cipher suites (TLS 1.0 and 1.1 have no such suites, so they cannot connect).
# Override TLS_CIPHERS only to match your organisation's cryptographic standard.
TLS_CIPHERS="${TLS_CIPHERS:-ECDHE+AESGCM:ECDHE+CHACHA20:!aNULL:!eNULL:!MD5:!SHA1}"
exec uvicorn app.main:app --host 0.0.0.0 --port "$PORT" --ssl-certfile "$CERT" --ssl-keyfile "$KEY" --ssl-ciphers "$TLS_CIPHERS" --no-server-header
