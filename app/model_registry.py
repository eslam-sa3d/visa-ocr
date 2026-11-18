"""Registry of OCR models available on this host.

Everything is local: models live in MODELS_DIR (default ./models) and are described by
models/manifest.json. Fine-tuned models are added with training/register_model.py, which
records a checksum and the evaluation metrics. The service refuses to load a model whose
files no longer match the recorded checksum, and (unless ALLOW_UNVERIFIED_MODELS=1) one with no checksum at all.

The manifest is always replaced whole (write a temp file, then rename), so a reader sees the old file or the new one, never
half of one. Writers take a lock file, so the retraining timer and an operator cannot lose each other's changes.

Optional signature: set MANIFEST_HMAC_KEY (or MANIFEST_HMAC_KEY_FILE) on the hosts that register models and on the hosts that
serve. The manifest then carries an HMAC-SHA256 over its model list, and a manifest whose signature is missing or wrong is
refused. Someone who can write the models folder but does not hold the key cannot swap a model and its checksum.
"""
from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path


class ManifestError(ValueError):
    """The manifest cannot be trusted or read: bad JSON, bad signature, or a model path outside the models folder."""


def default_models_dir() -> Path:
    return Path(os.environ.get("MODELS_DIR", Path(__file__).resolve().parent.parent / "models"))


def allow_unverified() -> bool:
    return os.environ.get("ALLOW_UNVERIFIED_MODELS", "").strip().lower() in ("1", "true", "yes", "on")


def _hmac_key() -> bytes | None:
    key = os.environ.get("MANIFEST_HMAC_KEY", "").strip()
    path = os.environ.get("MANIFEST_HMAC_KEY_FILE", "").strip()
    if not key and path:
        key = Path(path).read_text(encoding="utf-8").strip()
    return key.encode() if key else None


def _signature(models: list[dict], key: bytes) -> str:
    canonical = json.dumps(models, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hmac.new(key, canonical, hashlib.sha256).hexdigest()


def dir_sha256(path: Path) -> str:
    """Hash of a model directory: file names and contents, in sorted order."""
    h = hashlib.sha256()
    for f in sorted(p for p in Path(path).rglob("*") if p.is_file()):
        h.update(str(f.relative_to(path)).encode())
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def _dir_stamp(path: Path) -> tuple:
    """Names, sizes and modification times of a model directory: cheap to take, changes when a file is replaced."""
    out = []
    for f in sorted(p for p in Path(path).rglob("*") if p.is_file()):
        st = f.stat()
        out.append((str(f.relative_to(path)), st.st_size, st.st_mtime_ns))
    return tuple(out)


_HASH_CACHE: dict[str, tuple[tuple, str]] = {}      # folder -> (stamp, digest); used by status checks, never by a model load


@dataclass
class ModelInfo:
    id: str                      # unique id, for example ar_rec_ft_2026-12-01
    task: str                    # text_detection | text_recognition
    lang: str                    # en | ar | any
    version: str                 # pretrained | fine-tune version label
    path: str                    # folder under the models dir
    arch: str = ""               # PaddleOCR model name the weights belong to
    sha256: str | None = None
    active: bool = False
    source: str = ""
    metrics: dict = field(default_factory=dict)
    replaced: str = ""           # id of the model this one took over from; what --rollback returns to

    def architecture(self) -> str:
        return self.arch or self.id


_KNOWN = {f.name for f in fields(ModelInfo)}


class Registry:
    def __init__(self, base: Path | str | None = None) -> None:
        self.base = Path(base) if base else default_models_dir()
        self.manifest_path = self.base / "manifest.json"

    def models(self) -> list[ModelInfo]:
        if not self.manifest_path.exists():
            return []
        try:
            data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
            raw = data.get("models", [])
            key = _hmac_key()
            if key is not None and not hmac.compare_digest(str(data.get("signature", "")), _signature(raw, key)):
                raise ManifestError("the manifest signature is missing or wrong")
            return [ModelInfo(**{k: v for k, v in m.items() if k in _KNOWN}) for m in raw]
        except ManifestError:
            raise
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            raise ManifestError(f"the manifest cannot be read: {type(exc).__name__}") from exc

    def save(self, models: list[ModelInfo]) -> None:
        self.base.mkdir(parents=True, exist_ok=True)
        raw = [asdict(m) for m in models]
        payload: dict = {"models": raw}
        key = _hmac_key()
        if key is not None:
            payload["signature"] = _signature(raw, key)
        text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
        tmp = self.manifest_path.with_name(f"manifest.json.{os.getpid()}.tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(text)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, self.manifest_path)          # readers see the old file or the new one, never half of one

    @contextlib.contextmanager
    def locked(self):
        """One writer at a time: read, change and save the manifest inside this block."""
        self.base.mkdir(parents=True, exist_ok=True)
        with open(self.base / "manifest.lock", "w") as fh:
            try:
                import fcntl  # noqa: WPS433 - not on Windows; there the lock is skipped
                fcntl.flock(fh, fcntl.LOCK_EX)
            except ImportError:  # pragma: no cover
                pass
            yield

    def active(self, task: str, lang: str) -> ModelInfo | None:
        return next((m for m in self.models() if m.active and m.task == task and m.lang == lang), None)

    def full_path(self, m: ModelInfo) -> Path:
        """The model's folder. A path that leaves the models folder ('../x', an absolute path) is refused."""
        base = self.base.resolve()
        p = (self.base / m.path).resolve()
        if p == base or base not in p.parents:
            raise ManifestError(f"model {m.id} has a path outside the models folder")
        return p

    def status(self, m: ModelInfo, fresh: bool = True) -> str:
        """missing | unverified (no checksum recorded) | verified | mismatch.

        fresh=True hashes every file (always used before a model is loaded). fresh=False reuses the last digest while the
        folder's file names, sizes and times are unchanged: for status pages that are asked often."""
        try:
            p = self.full_path(m)
        except ManifestError:
            return "mismatch"
        if not p.is_dir():
            return "missing"
        if not m.sha256:
            return "unverified"
        if fresh:
            digest = dir_sha256(p)
            _HASH_CACHE[str(p)] = (_dir_stamp(p), digest)
        else:
            stamp = _dir_stamp(p)
            cached = _HASH_CACHE.get(str(p))
            if cached and cached[0] == stamp:
                digest = cached[1]
            else:
                digest = dir_sha256(p)
                _HASH_CACHE[str(p)] = (stamp, digest)
        return "verified" if digest == m.sha256 else "mismatch"

    def usable(self, m: ModelInfo, fresh: bool = True) -> tuple[bool, str]:
        """(may this model be loaded, its status). A model with no checksum is refused unless ALLOW_UNVERIFIED_MODELS=1."""
        st = self.status(m, fresh)
        return (st == "verified" or (st == "unverified" and allow_unverified())), st
