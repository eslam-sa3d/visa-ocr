#!/bin/sh
# Proves the built image starts and reads a document with NO network at all. Run it on the build host after `docker build`.
#   tools/offline_smoke.sh visa-ocr:latest path/to/a_passport.jpg
# The container gets no network interface (--network none), so any download, licence check or telemetry call fails here
# instead of on the air-gapped server. The request is made from inside the container for the same reason.
set -eu
IMAGE="${1:?image name}"; SAMPLE="${2:?a JPEG or PNG of a document}"
NAME="visa-ocr-offline-smoke-$$"
trap 'docker rm -f "$NAME" >/dev/null 2>&1 || true' EXIT
docker run -d --name "$NAME" --network none -e ALLOW_ANONYMOUS=1 -v "$(cd "$(dirname "$SAMPLE")" && pwd)/$(basename "$SAMPLE"):/tmp/sample:ro" "$IMAGE" >/dev/null
docker exec "$NAME" python - <<'PY'
import json, ssl, time, urllib.request, uuid
ctx = ssl._create_unverified_context()
base = "https://127.0.0.1:443"
for _ in range(90):
    try:
        urllib.request.urlopen(base + "/readyz", context=ctx, timeout=3); break
    except Exception:
        time.sleep(1)
else:
    raise SystemExit("FAIL: the service did not become ready with the network disabled")
data = open("/tmp/sample", "rb").read()
b = uuid.uuid4().hex
body = (f"--{b}\r\nContent-Disposition: form-data; name=\"files\"; filename=\"s.jpg\"\r\nContent-Type: image/jpeg\r\n\r\n").encode() + data + f"\r\n--{b}--\r\n".encode()
req = urllib.request.Request(base + "/v1/documents/extract", data=body, headers={"Content-Type": f"multipart/form-data; boundary={b}"})
job = json.load(urllib.request.urlopen(req, context=ctx, timeout=60))["jobId"]
for _ in range(120):
    j = json.load(urllib.request.urlopen(f"{base}/v1/jobs/{job}", context=ctx, timeout=10))
    if j["status"] in ("done", "failed"):
        break
    time.sleep(1)
print("status:", j["status"], "| engine:", (j.get("result") or {}).get("engine"), "| error:", (j.get("error") or {}).get("code"))
raise SystemExit(0 if j["status"] == "done" else "FAIL: the document was not read offline")
PY
echo "PASS: started and read a document with the network disabled"
