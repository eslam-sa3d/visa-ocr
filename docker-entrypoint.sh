#!/bin/sh
# Serves HTTPS on PORT (default 443). Use your own certificate: mount it and set TLS_CERT and TLS_KEY.
# Without one, a SELF-SIGNED certificate is generated so the service starts; browsers will warn. Test use only.
set -eu
umask 077        # files the service creates (database, TLS key, training data) are owner-only
PORT="${PORT:-443}"
# Same rule as app/security.py: a value made only of commas or spaces is not a key.
KEYS_SET="$(printf '%s%s' "${API_KEYS:-}" "${API_KEY_SHA256:-}" | tr -d ', \t\r\n')"
if [ -z "$KEYS_SET" ] && [ "${ALLOW_ANONYMOUS:-0}" != "1" ]; then
  echo "ERROR: no usable API key is configured (API_KEYS or API_KEY_SHA256). Set one, or set ALLOW_ANONYMOUS=1 for local tests only." >&2
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
  echo "WARNING: no TLS certificate at $CERT: generating a self-signed one (test only)" >&2
  mkdir -p "$(dirname "$CERT")"
  openssl req -x509 -newkey rsa:2048 -nodes -days 365 -subj "/CN=${TLS_CN:-localhost}" \
    -addext "subjectAltName=DNS:${TLS_CN:-localhost},DNS:localhost,IP:127.0.0.1" -keyout "$KEY" -out "$CERT" 2>/dev/null
  chmod 600 "$KEY"
fi
exec uvicorn app.main:app --host 0.0.0.0 --port "$PORT" --ssl-certfile "$CERT" --ssl-keyfile "$KEY"
