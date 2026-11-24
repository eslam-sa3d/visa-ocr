#!/usr/bin/env python3
"""Write or check SHA256SUMS for a folder that is carried to the offline server (the wheelhouse, models/tessdata, a golden set).

    python tools/make_checksums.py wheelhouse            # on the connected machine: writes wheelhouse/SHA256SUMS
    python tools/make_checksums.py --check wheelhouse    # on the offline server: every file must match, none may be extra

The file has the format of `sha256sum`, so `sha256sum -c SHA256SUMS` checks it too. Carry the SHA256SUMS file by a
different route from the folder (or sign it) if the transfer itself is not trusted. With MANIFEST_HMAC_KEY (or
MANIFEST_HMAC_KEY_FILE) set, SHA256SUMS.sig is written too: the service then uses models/tessdata only with a valid signature.
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

NAME = "SHA256SUMS"


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def files(folder: Path) -> list[Path]:
    return sorted(p for p in folder.rglob("*") if p.is_file() and p.name not in (NAME, NAME + ".sig"))


def write(folder: Path) -> int:
    rows = [f"{digest(p)}  {p.relative_to(folder).as_posix()}" for p in files(folder)]
    (folder / NAME).write_text("\n".join(rows) + "\n", encoding="utf-8")
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from app.model_registry import sign_bytes  # noqa: WPS433
    sig = sign_bytes((folder / NAME).read_bytes())          # with MANIFEST_HMAC_KEY set: models/tessdata needs this
    if sig:
        (folder / (NAME + ".sig")).write_text(sig + "\n", encoding="utf-8")
    return len(rows)


def check(folder: Path) -> list[str]:
    try:
        listed = dict(reversed(ln.split(None, 1)) for ln in (folder / NAME).read_text(encoding="utf-8").splitlines() if ln.strip())
    except OSError:
        return [f"{folder / NAME} is missing"]
    listed = {k.strip().lstrip("*"): v for k, v in listed.items()}
    problems = []
    present = {p.relative_to(folder).as_posix(): p for p in files(folder)}
    for name, want in listed.items():
        if name not in present:
            problems.append(f"missing: {name}")
        elif digest(present[name]) != want:
            problems.append(f"changed: {name}")
    problems += [f"not listed: {name}" for name in present if name not in listed]
    return problems


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", type=Path)
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    if a.check:
        problems = check(a.folder)
        for line in problems:
            print(line)
        print("OK" if not problems else f"{len(problems)} problem(s)")
        sys.exit(1 if problems else 0)
    print(f"wrote {a.folder / NAME} for {write(a.folder)} files")


if __name__ == "__main__":
    main()
