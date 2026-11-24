#!/bin/sh
# Proves the built image starts and works in the PRODUCTION set-up, with no network: PRODUCTION=1, keys with roles from a
# file, stored data encrypted, a mounted certificate, the audit log, a read-only image. Run it on the build host:
#   tools/production_smoke.sh visa-ocr:TAG path/to/a_passport.jpg
# It makes throw-away keys and a throw-away certificate in a temporary folder and removes them at the end.
# Checks: the service is ready; no key is refused; the portal key reads a document; the portal key may not list models;
# two reader processes start and one of them reads the document; a blank photo is turned back with a retake message; the stored result is encrypted; the audit log has lines.
set -eu
IMAGE="${1:?image name}"; SAMPLE="${2:?a JPEG or PNG of a document}"
HERE="$(cd "$(dirname "$0")" && pwd)"
NAME="visa-ocr-production-smoke-$$"
WORK="$(mktemp -d)"
trap 'code=$?; if [ $code -ne 0 ]; then echo "--- container output ---"; docker logs "$NAME" 2>&1 | tail -40; echo "--- container state: $(docker inspect -f "{{.State.Status}} exit={{.State.ExitCode}} oom={{.State.OOMKilled}}" "$NAME" 2>/dev/null) ---"; fi; docker rm -f "$NAME" >/dev/null 2>&1 || true; rm -rf "$WORK"; exit $code' EXIT
mkdir -p "$WORK/etc/tls" "$WORK/log"
python3 "$HERE/../deploy/production/make_keys.py" "$WORK/etc" > "$WORK/keys.txt"
PORTAL_KEY="$(awk '$1=="portal"{print $NF}' "$WORK/keys.txt")"
openssl req -x509 -newkey rsa:2048 -nodes -days 2 -subj "/CN=localhost" -addext "subjectAltName=DNS:localhost,IP:127.0.0.1" \
  -keyout "$WORK/etc/tls/server.key" -out "$WORK/etc/tls/server.crt" 2>/dev/null
chmod -R a+rX "$WORK/etc"; chmod a+r "$WORK/etc/keys.json" "$WORK/etc/data.key" "$WORK/etc/tls/server.key"; chmod a+rwx "$WORK/log"
docker run -d --name "$NAME" --network none --read-only --tmpfs /tmp --cap-drop ALL --security-opt no-new-privileges \
  -v "$WORK/etc:/etc/visa-ocr:ro" -v "$WORK/log:/var/log/visa-ocr" -v "$NAME-data:/data" \
  -v "$(cd "$(dirname "$SAMPLE")" && pwd)/$(basename "$SAMPLE"):/tmp/sample:ro" \
  -e PRODUCTION=1 -e API_KEYS_FILE=/etc/visa-ocr/keys.json -e DATA_KEY_FILE=/etc/visa-ocr/data.key -e DATA_CIPHER=aes256gcm \
  -e JOB_DB_PATH=/data/jobs.db -e TLS_CERT=/etc/visa-ocr/tls/server.crt -e TLS_KEY=/etc/visa-ocr/tls/server.key \
  -e AUDIT_LOG_FILE=/var/log/visa-ocr/audit.log -e RATE_LIMIT_PER_MIN=0 -e TRAINING_STORE_DIR=/data/training -e READER_PROCESSES=2 "$IMAGE" >/dev/null
OUT="$(docker exec -i -e PORTAL_KEY="$PORTAL_KEY" "$NAME" python - <<'PY'
import io, json, os, sqlite3, ssl, time, urllib.error, urllib.request, uuid
from PIL import Image
ctx = ssl._create_unverified_context(); base = "https://127.0.0.1:443"; key = {"X-API-Key": os.environ["PORTAL_KEY"]}
def call(path, data=None, headers=None):
    try:
        with urllib.request.urlopen(urllib.request.Request(base + path, data=data, headers=headers or {}), context=ctx, timeout=60) as r:
            return r.status, json.load(r)
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read() or b"{}")
        except ValueError:
            return e.code, {}
for _ in range(120):
    try:
        if call("/readyz")[0] == 200: break
    except Exception:
        time.sleep(1)
else:
    raise SystemExit("FAIL: the service did not become ready in the production set-up")
def upload(image):
    b = uuid.uuid4().hex
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"files\"; filename=\"s.png\"\r\nContent-Type: image/png\r\n\r\n").encode() + image + f"\r\n--{b}--\r\n".encode()
    return body, {"Content-Type": f"multipart/form-data; boundary={b}"}
def read(image):
    return read_at("/v1/documents/extract", image)
def read_at(path, image):
    body, h = upload(image)
    status, j = call(path, body, {**h, **key})
    if status == 422 and (j.get("detail") or {}).get("code") == "image_unclear":
        return {"status": "failed", "error": j["detail"], "refusedAtUpload": True}     # refused at once, no job
    assert status == 202, f"upload gave {status}"
    for _ in range(420):
        j = call("/v1/jobs/" + j["jobId"], headers=key)[1]
        if j.get("status") in ("done", "failed"): return j
        time.sleep(1)
    raise SystemExit("FAIL: the job did not finish")
sample = open("/tmp/sample", "rb").read()
body, h = upload(sample)
assert call("/v1/documents/extract", body, h)[0] == 401, "an upload with no key was not refused"
assert call("/ui")[0] == 404, "the test page is on in production"
assert call("/v1/models", headers=key)[0] == 403, "the portal key may list models"
done = read(sample)
blank = io.BytesIO(); Image.new("RGB", (1200, 900), "white").save(blank, format="PNG")
turned_back = read(blank.getvalue())
retake = ((turned_back.get("error") or {}).get("retake") or {}).get("message")
rows = sqlite3.connect("/data/jobs.db").execute("select * from jobs").fetchall()
sealed = any(str(c).startswith("VOENC") for r in rows for c in r)
plain = any((done.get("result") or {}).get("document", {}).get("documentNumber", "\0") in str(c) for r in rows for c in r)
audit = sum(1 for _ in open("/var/log/visa-ocr/audit.log"))
pb = uuid.uuid4().hex                           # personal-photo check: the face detector must be inside the image
pbody = (f"--{pb}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"me.png\"\r\nContent-Type: image/png\r\n\r\n").encode() + blank.getvalue() + f"\r\n--{pb}--\r\n".encode()
pstatus, pverdict = call("/v1/photos/check", pbody, {"Content-Type": f"multipart/form-data; boundary={pb}", **key})
photo_ok = pstatus == 200 and pverdict.get("acceptable") is False and "no_face" in (pverdict.get("issues") or [])
pjob = read_at("/v1/passports/extract", sample)   # passport-only endpoint reads the sample passport
readers = call("/readyz")[1].get("readers")
print("status:", done["status"], "| engine:", (done.get("result") or {}).get("engine"), "| readers:", readers, "| passport endpoint:", pjob.get("status"), "| photo check:", "ok" if photo_ok else f"FAILED ({pstatus})", "| blank photo:", (turned_back.get("error") or {}).get("code"),
      "| retake message:", bool(retake), "| stored encrypted:", sealed and not plain, "| audit lines:", audit)
ok = readers == 2 and photo_ok and pjob.get("status") == "done" and done["status"] == "done" and turned_back["status"] == "failed" and retake and sealed and not plain and audit > 0
raise SystemExit(0 if ok else "FAIL: see the line above")
PY
)"
echo "$OUT"
echo "$OUT" | grep -q "^status: done" || { echo "FAIL: no finished read was reported from inside the container" >&2; exit 1; }
docker volume rm "$NAME-data" >/dev/null 2>&1 || true
echo "PASS: production set-up started, read a document, refused what it should, and stored the result encrypted"
