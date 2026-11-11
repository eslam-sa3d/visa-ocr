"""API-key authentication, rate limiting and security headers.

Set API_KEYS (comma-separated secrets) or API_KEY_SHA256 (comma-separated SHA-256 hex digests) to require a key
in the X-API-Key header (or 'Authorization: Bearer <key>'). With neither set, authentication is OFF: development only.
Rate limiting is per key (or per client address) and per process; put a gateway in front for a global limit.
"""
from __future__ import annotations

import hashlib
import hmac
import os
import re
import threading
import time
from collections import defaultdict, deque

from fastapi import HTTPException, Request

from . import config

_CID = re.compile(r"[A-Za-z0-9._\-]{1,64}")


def _digests() -> list[str]:
    raw_keys, raw_hashes = os.environ.get("API_KEYS", ""), os.environ.get("API_KEY_SHA256", "")
    out = [hashlib.sha256(k.strip().encode()).hexdigest() for k in raw_keys.split(",") if k.strip()]
    out += [h.strip().lower() for h in raw_hashes.split(",") if h.strip()]
    if not out and (raw_keys or raw_hashes):
        # A key variable is set but holds no usable key (for example "," from two empty variables). Fail closed:
        # treating this as "no key configured" would silently turn authentication off.
        raise RuntimeError("API_KEYS / API_KEY_SHA256 is set but contains no usable key")
    return out


def weak_plain_keys() -> int:
    """How many plain API_KEYS values are shorter than 24 characters (guessable keys)."""
    return sum(1 for k in os.environ.get("API_KEYS", "").split(",") if k.strip() and len(k.strip()) < 24)


def auth_enabled() -> bool:
    return bool(_digests())


def authenticate(request: Request) -> str:
    """FastAPI dependency: returns the caller's owner id (a short hash of the key) or raises 401."""
    digests = _digests()
    if not digests:
        return "anonymous"
    key = request.headers.get("X-API-Key", "")
    auth = request.headers.get("Authorization", "")
    if not key and auth.lower().startswith("bearer "):
        key = auth[7:].strip()
    given = hashlib.sha256(key.encode()).hexdigest()
    for d in digests:
        if hmac.compare_digest(given, d):
            return given[:16]
    # Throttle wrong keys per client address so a key cannot be guessed at full speed
    limiter.check("badkey:" + client_id(request, "anonymous"), limit=config.bad_key_limit_per_minute())
    raise HTTPException(401, {"code": "unauthorized", "message": "A valid API key is required"},
                        headers={"WWW-Authenticate": "ApiKey"})


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


class BodyLimit:
    """Pure ASGI guard: reject an oversized request body before the framework buffers it (and before auth runs)."""

    def __init__(self, app) -> None:
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or not scope.get("path", "").startswith("/v1"):
            return await self.app(scope, receive, send)
        limit = config.max_body_bytes()
        declared = dict(scope.get("headers") or []).get(b"content-length")
        if declared and declared.isdigit() and int(declared) > limit:
            return await self._reject(send)
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
