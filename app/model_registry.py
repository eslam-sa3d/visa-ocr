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

The signed data includes a serial number that every save increases. A server remembers the highest serial it has accepted
for each manifest (in memory, and in MANIFEST_STATE_FILE when that is set: put it on the data volume, outside the models
folder) and refuses an older manifest put back in place, so an earlier signed state (with a model since replaced) cannot be
restored by someone without the key. With the key set, SHA256SUMS in models/tessdata must carry SHA256SUMS.sig (written by
tools/make_checksums.py), so the Tesseract MRZ model is under the same key.
"""
from __future__ import annotations

import contextlib
import hashlib
import hmac
import json
import os
import re
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


def _signature(models: list[dict], key: bytes, serial: int | None = None) -> str:
    body = models if serial is None else {"models": models, "serial": serial}      # None: manifests written before serials
    canonical = json.dumps(body, sort_keys=True, ensure_ascii=False, separators=(",", ":")).encode()
    return hmac.new(key, canonical, hashlib.sha256).hexdigest()


def sign_bytes(data: bytes) -> str | None:
    """HMAC-SHA256 of a file's bytes with the manifest key, or None when no key is set."""
    key = _hmac_key()
    return hmac.new(key, data, hashlib.sha256).hexdigest() if key else None


def signed_file_ok(path: Path) -> bool:
    """With a manifest key set: the file's '<name>.sig' holds its HMAC. Without a key there is nothing to check."""
    key = _hmac_key()
    if key is None:
        return True
    try:
        data = Path(path).read_bytes()
        sig = Path(str(path) + ".sig").read_text(encoding="utf-8").strip()
    except OSError:
        return False
    return hmac.compare_digest(sig, hmac.new(key, data, hashlib.sha256).hexdigest())


_HIGHEST: dict[str, int] = {}          # manifest path -> highest serial accepted by this process


def _state_file() -> Path | None:
    p = os.environ.get("MANIFEST_STATE_FILE", "").strip()
    return Path(p) if p else None


def _highest_seen(manifest: Path) -> int:
    key = str(manifest.resolve())
    best = _HIGHEST.get(key, -1)
    sf = _state_file()
    if sf is not None:
        try:
            best = max(best, int(json.loads(sf.read_text(encoding="utf-8")).get(key, -1)))
        except (OSError, ValueError, TypeError, AttributeError):
            pass
    return best


def _remember(manifest: Path, serial: int) -> None:
    key = str(manifest.resolve())
    if serial <= _HIGHEST.get(key, -1):
        return
    _HIGHEST[key] = serial
    sf = _state_file()
    if sf is None:
        return
    try:
        state = json.loads(sf.read_text(encoding="utf-8")) if sf.exists() else {}
        if int(state.get(key, -1)) >= serial:
            return
        state[key] = serial
        tmp = sf.with_name(sf.name + f".{os.getpid()}.tmp")
        tmp.write_text(json.dumps(state), encoding="utf-8")
        os.replace(tmp, sf)
    except (OSError, ValueError, TypeError):
        pass


_DESKTOP_FILES = {".DS_Store", "Thumbs.db", "desktop.ini"}      # written by file managers when a folder is opened; not model files


def _model_files(path: Path) -> list[Path]:
    return sorted(p for p in Path(path).rglob("*") if p.is_file() and p.name not in _DESKTOP_FILES)


def dir_sha256(path: Path) -> str:
    """Hash of a model directory: file names and contents, in sorted order. Files a desktop file manager drops into a folder
    (.DS_Store and the like) are left out, so looking at a model folder in Finder does not make it fail its checksum."""
    h = hashlib.sha256()
    for f in _model_files(path):
        h.update(str(f.relative_to(path)).encode())
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def _dir_stamp(path: Path) -> tuple:
    """Names, sizes and modification times of a model directory: cheap to take, changes when a file is replaced."""
    out = []
    for f in _model_files(path):
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
_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,99}")


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
            serial = data.get("serial")
            if serial is not None and (not isinstance(serial, int) or isinstance(serial, bool) or serial < 0):
                raise ManifestError("the manifest serial is not a whole number")
            key = _hmac_key()
            if key is not None:
                if not hmac.compare_digest(str(data.get("signature", "")), _signature(raw, key, serial)):
                    raise ManifestError("the manifest signature is missing or wrong")
                seen = _highest_seen(self.manifest_path)
                if (serial or 0) < seen:
                    raise ManifestError(f"an older manifest was put back (serial {serial or 0}; {seen} was accepted before)")
                _remember(self.manifest_path, serial or 0)
            out = [ModelInfo(**{k: v for k, v in m.items() if k in _KNOWN}) for m in raw]
            for m in out:
                # ids name folders and files on the training host (checkpoints, metrics): letters, digits, '.', '_' and '-'
                # only, so an id can never point outside the folder it is used in
                if not isinstance(m.id, str) or not _ID.fullmatch(m.id) or ".." in m.id:
                    raise ManifestError(f"a model id is not allowed: {str(m.id)[:60]!r}")
            return out
        except ManifestError:
            raise
        except (OSError, ValueError, TypeError, AttributeError) as exc:
            raise ManifestError(f"the manifest cannot be read: {type(exc).__name__}") from exc

    def save(self, models: list[ModelInfo]) -> None:
        self.base.mkdir(parents=True, exist_ok=True)
        raw = [asdict(m) for m in models]
        serial = 0
        try:
            serial = int(json.loads(self.manifest_path.read_text(encoding="utf-8")).get("serial") or 0) + 1
        except (OSError, ValueError, TypeError, AttributeError):
            serial = 1
        payload: dict = {"serial": serial, "models": raw}
        key = _hmac_key()
        if key is not None:
            payload["signature"] = _signature(raw, key, serial)
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
