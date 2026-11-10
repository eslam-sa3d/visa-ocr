#!/bin/sh
# Serves HTTPS on PORT (default 443). Use your own certificate: mount it and set TLS_CERT and TLS_KEY.
# Without one, a SELF-SIGNED certificate is generated so the service starts; browsers will warn. Test use only.
set -eu
umask 077        # files the service creates (database, TLS key, training data) are owner-only
PORT="${PORT:-443}"
if [ -z "${API_KEYS:-}" ] && [ -z "${API_KEY_SHA256:-}" ] && [ "${ALLOW_ANONYMOUS:-0}" != "1" ]; then
  echo "ERROR: no API key is configured (API_KEYS or API_KEY_SHA256). Set one, or set ALLOW_ANONYMOUS=1 for local tests only." >&2
  exit 1
fi
CERT="${TLS_CERT:-/data/tls/server.crt}"
KEY="${TLS_KEY:-/data/tls/server.key}"
if [ ! -s "$CERT" ] || [ ! -s "$KEY" ]; then
  echo "WARNING: no TLS certificate at $CERT: generating a self-signed one (test only)" >&2
  mkdir -p "$(dirname "$CERT")"
  openssl req -x509 -newkey rsa:2048 -nodes -days 365 -subj "/CN=${TLS_CN:-localhost}" \
    -addext "subjectAltName=DNS:${TLS_CN:-localhost},DNS:localhost,IP:127.0.0.1" -keyout "$KEY" -out "$CERT" 2>/dev/null
  chmod 600 "$KEY"
fi
exec uvicorn app.main:app --host 0.0.0.0 --port "$PORT" --ssl-certfile "$CERT" --ssl-keyfile "$KEY"
