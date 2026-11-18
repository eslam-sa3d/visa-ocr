#!/usr/bin/env python3
"""Bill of materials and vulnerability check for the locked packages (NCA ECC 1-6-3-2 and 2-10-3).

    python tools/dependency_audit.py                      # writes docs/sbom.json from requirements.lock
    python tools/dependency_audit.py --lock requirements-lite.lock --out docs/sbom-lite.json
    python tools/dependency_audit.py --check              # also checks every package for known vulnerabilities

The bill of materials lists every package, direct and indirect, with its exact version, how many file hashes pin it, and its
licence as declared in the package itself (read from the environment this runs in; "not installed here" when a package is
not present, so run it in the environment built from the lock).

--check needs the `pip-audit` package and a connection to the Python advisory database, so run it on the connected build
machine, at every release and on a schedule (weekly is usual). It exits 1 when a known vulnerability is found. The offline
server needs neither: what is installed there is exactly what was checked here, because the lock pins every file by hash.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def locked(path: Path) -> list[dict]:
    out, current = [], None
    for line in path.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^([A-Za-z0-9_.\-]+)==([^\s\\]+)", line)
        if m:
            current = {"name": m[1], "version": m[2], "hashes": 0}
            out.append(current)
        elif current is not None and "--hash=sha256:" in line:
            current["hashes"] += 1
    return out


def licence(name: str) -> str:
    try:
        meta = metadata.metadata(name)
    except metadata.PackageNotFoundError:
        return "not installed here"
    value = (meta.get("License-Expression") or meta.get("License") or "").strip().splitlines()
    if value and len(value[0]) <= 80:
        return value[0]
    classifiers = [c.split("::")[-1].strip() for c in meta.get_all("Classifier") or [] if c.startswith("License ::")]
    return "; ".join(classifiers) or "not declared"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--lock", type=Path, default=ROOT / "requirements.lock")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "sbom.json")
    ap.add_argument("--check", action="store_true", help="look every package up in the advisory database (needs pip-audit and a connection)")
    a = ap.parse_args()
    packages = locked(a.lock)
    for p in packages:
        p["licence"] = licence(p["name"])
    sbom = {"generatedAt": datetime.now(timezone.utc).isoformat(timespec="seconds"), "lockFile": a.lock.name,
            "python": sys.version.split()[0], "packages": packages,
            "unpinned": [p["name"] for p in packages if not p["hashes"]]}
    found = None
    if a.check:
        report = a.out.with_suffix(".audit.json")
        run = subprocess.run([sys.executable, "-m", "pip_audit", "-r", str(a.lock), "--require-hashes", "--disable-pip",
                              "-f", "json", "-o", str(report)], capture_output=True, text=True)
        if not report.exists():
            sys.exit("the vulnerability check did not run: " + (run.stderr.strip().splitlines() or ["pip-audit is not installed"])[-1])
        deps = json.loads(report.read_text(encoding="utf-8")).get("dependencies", [])
        found = [{"name": d["name"], "version": d["version"], "ids": [v["id"] for v in d["vulns"]],
                  "fixedIn": sorted({f for v in d["vulns"] for f in v.get("fix_versions", [])})} for d in deps if d.get("vulns")]
        sbom["vulnerabilityCheck"] = {"checkedAt": sbom["generatedAt"], "packagesChecked": len(deps), "found": found}
        report.unlink()
    a.out.write_text(json.dumps(sbom, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"{len(packages)} packages written to {a.out}" + (f"; unpinned: {sbom['unpinned']}" if sbom["unpinned"] else "; every one pinned by hash"))
    if found is not None:
        print("known vulnerabilities:", found if found else "none")
        sys.exit(1 if found else 0)


if __name__ == "__main__":
    main()
