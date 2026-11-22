#!/usr/bin/env python3
"""Create the secrets a production install needs, in one folder (default /etc/visa-ocr):

    keys.json   API keys as SHA-256 digests with a role and an expiry date (the keys themselves are printed ONCE, here)
    data.key    the key that encrypts stored personal data (AES-256-GCM)

    sudo python3 make_keys.py /etc/visa-ocr [--expires 2027-11-30] [--owner 999]

Nothing is overwritten: run it in an empty folder, or move the old files away first (keep the old data.key until the data
encrypted with it has expired). Needs only Python 3; no packages.
"""
import argparse
import base64
import hashlib
import json
import os
import secrets
from datetime import date, timedelta
from pathlib import Path

ROLES = (("portal", "uploader"), ("review", "reviewer"), ("ops", "admin"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("folder", type=Path, nargs="?", default=Path("/etc/visa-ocr"))
    ap.add_argument("--expires", default=(date.today() + timedelta(days=365)).isoformat(), help="last valid day of the keys")
    ap.add_argument("--owner", type=int, help="user id that must be able to read the files (the service user in the image)")
    a = ap.parse_args()
    a.folder.mkdir(parents=True, exist_ok=True)
    keys_file, data_key = a.folder / "keys.json", a.folder / "data.key"
    for f in (keys_file, data_key):
        if f.exists():
            raise SystemExit(f"{f} already exists; nothing was changed")
    entries, shown = [], []
    for name, role in ROLES:
        key = secrets.token_hex(32)
        entries.append({"name": name, "sha256": hashlib.sha256(key.encode()).hexdigest(), "role": role, "expires": a.expires})
        shown.append((name, role, key))
    for path, text in ((keys_file, json.dumps(entries, indent=1) + "\n"),
                       (data_key, base64.urlsafe_b64encode(os.urandom(32)).decode() + "\n")):
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o400)
        with os.fdopen(fd, "w") as fh:
            fh.write(text)
        if a.owner is not None:
            os.chown(path, a.owner, a.owner)
    print(f"Written: {keys_file} and {data_key} (read-only). Keys expire after {a.expires}.")
    print("Give each key to its owner now and store it in your key management system. They are not saved anywhere else:\n")
    for name, role, key in shown:
        print(f"  {name:<7} role {role:<9} X-API-Key: {key}")
    print("\nportal: sends documents and reads its own results.  review: corrections.  ops: models and maintenance.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
