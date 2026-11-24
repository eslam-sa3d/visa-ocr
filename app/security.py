"""API-key authentication, roles, rate limiting and security headers.

Set API_KEYS (comma-separated secrets) or API_KEY_SHA256 (comma-separated SHA-256 hex digests) to require a key
in the X-API-Key header (or 'Authorization: Bearer <key>'). With neither set, authentication is OFF: development only.
Rate limiting is per key (or per client address) and per process; put a gateway in front for a global limit.

Roles (least privilege, NCA ECC 2-2-3-3). Keys given in API_KEYS / API_KEY_SHA256 have the role "full": every action, on
their own jobs. For separate duties list the keys in a JSON file and set API_KEYS_FILE to it:

    [{"name": "portal",    "sha256": "<hex digest of the key>", "role": "uploader", "expires": "2027-06-30"},
     {"name": "review-ui", "sha256": "<hex digest>",            "role": "reviewer"},
     {"name": "ops",       "sha256": "<hex digest>",            "role": "admin", "expires": "2027-01-31"}]

    uploader   upload documents, read its own jobs, parse MRZ text
    reviewer   read any job and submit corrections (it cannot upload)
    admin      list models and their status, read any job
    full       everything above on its own jobs (the single-key set-up)

A key past its "expires" date (inclusive, UTC) is refused. Make a digest with:  printf %s "$KEY" | sha256sum
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import threading
import time
from datetime import date, datetime, timezone
from collections import defaultdict, deque

from fastapi import HTTPException, Request

from . import audit, config

_CID = re.compile(r"[A-Za-z0-9._\-]{1,64}")


PERMISSIONS = {
    "uploader": {"upload", "read", "parse"},
    "reviewer": {"read", "read_any", "correct", "parse"},
    "admin": {"read", "read_any", "models"},
    "full": {"upload", "read", "parse", "correct", "models"},
}
_file_cache: tuple[str, float, list[dict]] | None = None


def _file_keys() -> list[dict]:
    """Keys with a role and an optional expiry date from API_KEYS_FILE. A file that is set but unusable stops the service:
    falling back to "no keys" would turn authentication off."""
    global _file_cache
    path = os.environ.get("API_KEYS_FILE", "").strip()
    if not path:
        return []
    try:
        mtime = os.stat(path).st_mtime
        if _file_cache and _file_cache[0] == path and _file_cache[1] == mtime:
            return _file_cache[2]
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
        keys = []
        for item in raw:
            digest, role = str(item["sha256"]).strip().lower(), str(item.get("role", "")).strip()
            if not re.fullmatch(r"[0-9a-f]{64}", digest) or role not in PERMISSIONS:
                raise ValueError("each entry needs a 64-character sha256 and a role of " + ", ".join(sorted(PERMISSIONS)))
            expires = date.fromisoformat(item["expires"]) if item.get("expires") else None
            keys.append({"digest": digest, "role": role, "expires": expires, "name": str(item.get("name", ""))[:64]})
        if not keys:
            raise ValueError("no keys listed")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        raise RuntimeError(f"API_KEYS_FILE cannot be used: {type(exc).__name__}: {exc}") from exc
    _file_cache = (path, mtime, keys)
    return keys


def _keys() -> list[dict]:
    raw_keys, raw_hashes = os.environ.get("API_KEYS", ""), os.environ.get("API_KEY_SHA256", "")
    out = [hashlib.sha256(k.strip().encode()).hexdigest() for k in raw_keys.split(",") if k.strip()]
    out += [h.strip().lower() for h in raw_hashes.split(",") if h.strip()]
    if not out and (raw_keys or raw_hashes):
        # A key variable is set but holds no usable key (for example "," from two empty variables). Fail closed:
        # treating this as "no key configured" would silently turn authentication off.
        raise RuntimeError("API_KEYS / API_KEY_SHA256 is set but contains no usable key")
    return [{"digest": d, "role": "full", "expires": None, "name": ""} for d in out] + _file_keys()


def _digests() -> list[str]:
    return [k["digest"] for k in _keys()]


def weak_plain_keys() -> int:
    """How many plain API_KEYS values are shorter than 24 characters (guessable keys)."""
    return sum(1 for k in os.environ.get("API_KEYS", "").split(",") if k.strip() and len(k.strip()) < 24)


def auth_enabled() -> bool:
    return bool(_digests())


def authenticate(request: Request) -> str:
    """FastAPI dependency: returns the caller's owner id (a short hash of the key) or raises 401."""
    keys = _keys()
    request.state.role = "full"
    if not keys:
        return "anonymous"
    key = request.headers.get("X-API-Key", "")
    auth = request.headers.get("Authorization", "")
    if not key and auth.lower().startswith("bearer "):
        key = auth[7:].strip()
    given = hashlib.sha256(key.encode()).hexdigest()
    match = None
    for k in keys:                                   # every entry is compared, so timing does not reveal which one matched
        if hmac.compare_digest(given, k["digest"]):
            match = match or k
    path = request.url.path
    if match is not None:
        if match["expires"] is not None and datetime.now(timezone.utc).date() > match["expires"]:
            audit.event("auth_key_expired", key=given[:16], key_name=match["name"], client=client_id(request, "anonymous"), path=path)
            raise HTTPException(401, {"code": "key_expired", "message": "This API key has expired"},
                                headers={"WWW-Authenticate": "ApiKey"})
        request.state.role = match["role"]
        return given[:16]
    # Throttle wrong keys per client address so a key cannot be guessed at full speed
    client = client_id(request, "anonymous")
    try:
        limiter.check("badkey:" + client, limit=config.bad_key_limit_per_minute())
    except HTTPException:
        if audit.first_in_window("auth_throttled:" + client):   # once per client per minute, not once per request
            audit.event("auth_throttled", client=client, path=path)
        raise
    audit.event("auth_failed", client=client, path=path, key_sent=bool(key))
    raise HTTPException(401, {"code": "unauthorized", "message": "A valid API key is required"},
                        headers={"WWW-Authenticate": "ApiKey"})


def allow(request: Request, permission: str, owner: str = "") -> str:
    """Refuse (403) unless the caller's role includes `permission`. Returns the role."""
    role = getattr(request.state, "role", "full")
    if permission not in PERMISSIONS.get(role, ()):
        audit.event("forbidden", key=owner, role=role, permission=permission, path=request.url.path)
        raise HTTPException(403, {"code": "forbidden", "message": f"This API key's role ({role}) does not allow this action"})
    return role


def may_read_any(request: Request) -> bool:
    return "read_any" in PERMISSIONS.get(getattr(request.state, "role", "full"), ())


class RateLimiter:
    def __init__(self) -> None:
        self._hits: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, who: str, limit: int | None = None, window: float = 60.0) -> None:
        limit = config.rate_limit_per_minute() if limit is None else limit
        if limit <= 0:
            return
        now = time.monotonic()
        with self._lock:
            q = self._hits[who]
            while q and q[0] <= now - window:
                q.popleft()
            if len(q) >= limit:
                retry = max(1, int(window - (now - q[0])))
                raise HTTPException(429, {"code": "rate_limited", "message": "Too many requests; slow down"},
                                    headers={"Retry-After": str(retry)})
            q.append(now)
            if len(self._hits) > 10_000:                      # forget idle callers so the table cannot grow without bound
                for k in [k for k, v in self._hits.items() if not v or v[-1] <= now - window]:
                    del self._hits[k]


limiter = RateLimiter()


def client_id(request: Request, owner: str) -> str:
    if owner != "anonymous":
        return owner
    if config.trust_proxy():
        fwd = request.headers.get("X-Forwarded-For", "")
        if fwd:
            return fwd.split(",")[-1].strip()           # the hop your own proxy appended; earlier hops are client-chosen
    return request.client.host if request.client else "unknown"


def safe_correlation_id(value: str | None, fallback: str) -> str:
    return value if value and _CID.fullmatch(value) else fallback


UI_CSP = ("default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; "
          "img-src 'self' blob: data:; connect-src 'self'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'")


def add_security_headers(response, path: str) -> None:
    response.headers.setdefault("X-Content-Type-Options", "nosniff")
    response.headers.setdefault("Referrer-Policy", "no-referrer")
    response.headers.setdefault("X-Frame-Options", "DENY")
    if path.startswith("/v1"):
        response.headers.setdefault("Cache-Control", "no-store")      # results are personal data
    if path == "/ui":
        response.headers.setdefault("Content-Security-Policy", UI_CSP)


def refuse_before_body(scope) -> tuple[int, dict, dict] | None:
    """For a request that carries a body: (status, detail, headers) when its API key is missing or unknown, so the body is
    never read or parsed for a caller without a key. Refusals are throttled and recorded exactly as authenticate() does. A
    known key (also an expired one, which authenticate() answers) returns None and the request goes on."""
    from starlette.requests import Request as _Request  # noqa: WPS433
    keys = _keys()
    if not keys:
        return None
    request = _Request(scope)
    key = request.headers.get("X-API-Key", "")
    auth = request.headers.get("Authorization", "")
    if not key and auth.lower().startswith("bearer "):
        key = auth[7:].strip()
    given = hashlib.sha256(key.encode()).hexdigest()
    known = False
    for k in keys:                                   # every entry is compared, as in authenticate()
        known = hmac.compare_digest(given, k["digest"]) or known
    if known:
        return None
    client = client_id(request, "anonymous")
    path = request.url.path
    try:
        limiter.check("badkey:" + client, limit=config.bad_key_limit_per_minute())
    except HTTPException as exc:
        if audit.first_in_window("auth_throttled:" + client):
            audit.event("auth_throttled", client=client, path=path)
        return exc.status_code, exc.detail, dict(exc.headers or {})
    audit.event("auth_failed", client=client, path=path, key_sent=bool(key))
    return 401, {"code": "unauthorized", "message": "A valid API key is required"}, {"WWW-Authenticate": "ApiKey"}


class BodyLimit:
    """Pure ASGI guard: reject an oversized request body before the framework buffers it (and before auth runs), and refuse
    a request with a body and no valid API key before any of the body is read."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not scope.get("path", "").startswith("/v1"):
            return await self.app(scope, receive, send)
        limit = config.max_body_bytes()
        declared = dict(scope.get("headers") or []).get(b"content-length")
        if declared and declared.isdigit() and int(declared) > limit:
            return await self._reject(send)
        if scope.get("method") in ("POST", "PUT", "PATCH"):
            refused = refuse_before_body(scope)
            if refused is not None:
                import json as _json  # noqa: WPS433
                status, detail, headers = refused
                body = _json.dumps({"detail": detail}).encode()
                await send({"type": "http.response.start", "status": status,
                            "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode()),
                                        (b"connection", b"close")]
                            + [(k.lower().encode(), str(v).encode()) for k, v in headers.items()]})
                await send({"type": "http.response.body", "body": body})
                return
        seen = 0
        started = False
        exceeded = False
        rejected = False

        async def counting_receive():
            nonlocal seen, exceeded
            msg = await receive()
            if msg["type"] == "http.request":
                seen += len(msg.get("body", b""))
                if seen > limit:
                    exceeded = True
                    raise _TooLarge()
            return msg

        async def tracking_send(msg):
            nonlocal started, rejected
            if exceeded and not started:
                # The form parser caught the error and is about to answer "400 could not parse": say what really happened.
                if not rejected:
                    rejected = True
                    await self._reject(send)
                return
            if msg["type"] == "http.response.start":
                started = True
            await send(msg)

        try:
            await self.app(scope, counting_receive, tracking_send)
        except _TooLarge:
            if not started and not rejected:
                await self._reject(send)

    @staticmethod
    async def _reject(send) -> None:
        body = b'{"detail":{"code":"body_too_large","message":"The request body is too large"}}'
        await send({"type": "http.response.start", "status": 413,
                    "headers": [(b"content-type", b"application/json"), (b"content-length", str(len(body)).encode()),
                                (b"connection", b"close")]})
        await send({"type": "http.response.body", "body": body})


class _TooLarge(Exception):
    pass
