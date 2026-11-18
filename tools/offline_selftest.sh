#!/bin/sh
# Proves, WITHOUT Docker, that the service starts and reads documents with no network: everything runs inside a new network
# namespace that has only the loopback interface, so any download, licence check, telemetry call or DNS lookup fails.
# Needs root (or CAP_SYS_ADMIN) for `unshare -n`. Run from the project folder:   sudo tools/offline_selftest.sh
# It first shows that the outside world really is unreachable, then runs the functional system test against the service.
set -eu
PORT="${PORT:-8791}"
PY="${PYTHON:-python3}"
exec unshare -n sh -c '
  set -eu
  # bring the loopback interface up (ip, ifconfig, or directly when neither tool is installed)
  ip link set lo up 2>/dev/null || ifconfig lo up 2>/dev/null || '"$PY"' -c "
import fcntl, socket, struct
s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
flags = struct.unpack(\"16sH\", fcntl.ioctl(s, 0x8913, struct.pack(\"16sH\", b\"lo\", 0)))[1]
fcntl.ioctl(s, 0x8914, struct.pack(\"16sH\", b\"lo\", flags | 1))"
  unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy   # a proxy setting would send even 127.0.0.1 outside
  export NO_PROXY=127.0.0.1,localhost
  if '"$PY"' -c "import socket,sys; s=socket.socket(); s.settimeout(3); s.connect((\"1.1.1.1\",443))" 2>/dev/null; then
    echo "FAIL: the network is reachable from inside the namespace; this run would prove nothing" >&2; exit 2
  fi
  echo "network unreachable inside the namespace: good"
  ALLOW_ANONYMOUS=1 RATE_LIMIT_PER_MIN=0 '"$PY"' -m uvicorn app.main:app --host 127.0.0.1 --port '"$PORT"' >/tmp/offline_selftest_server.log 2>&1 &
  SERVER=$!
  trap "kill $SERVER 2>/dev/null || true" EXIT
  i=0; until '"$PY"' -c "import urllib.request as u; u.urlopen(\"http://127.0.0.1:'"$PORT"'/readyz\", timeout=2)" 2>/dev/null; do
    i=$((i+1)); [ $i -lt 60 ] || { echo "FAIL: the service did not become ready offline" >&2; tail -20 /tmp/offline_selftest_server.log >&2; exit 1; }
    sleep 1
  done
  '"$PY"' tools/system_test.py --url http://127.0.0.1:'"$PORT"' --only functional
'
