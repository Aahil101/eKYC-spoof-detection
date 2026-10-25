"""Package the repo source as a Kaggle dataset (for internet-OFF notebooks).

    python -m training.package_src --owner <your-kaggle-username>
    kaggle datasets create -p dist/codmav-src        # first time
    kaggle datasets version -p dist/codmav-src -m "update"   # afterwards

Excludes data, media, runs, models/store, node_modules, secrets. Writes GIT_SHA into the
zip so runs record which code produced them even without .git.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

from training.paths import REPO_ROOT

INCLUDE_DIRS = ("backend", "ml", "training", "monitoring", "experiments", "config", "requirements")
INCLUDE_FILES = ("pyproject.toml", "README.md")
EXCLUDE_PARTS = {"__pycache__", ".pytest_cache", "node_modules", "runs", "results", "build"}
EXCLUDE_SUFFIX = {".pyc", ".db", ".sqlite", ".bin", ".pt", ".pkl", ".npz", ".keys"}


def iter_files(root: Path):
    for d in INCLUDE_DIRS:
        base = root / d
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if p.is_file() and not (set(p.relative_to(root).parts) & EXCLUDE_PARTS) and p.suffix not in EXCLUDE_SUFFIX:
                yield p
    for f in INCLUDE_FILES:
        if (root / f).exists():
            yield root / f


def build(owner: str, out_dir: Path, root: Path = REPO_ROOT, slug: str = "codmav-src") -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(iter_files(root), key=lambda p: p.relative_to(root).as_posix())
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True,
                                      stderr=subprocess.DEVNULL).strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True,
                                        stderr=subprocess.DEVNULL).strip()
        if dirty:
            sha += "-dirty"
    except Exception:
        sha = ""
    # Content hash makes runs traceable to the exact code even without git (or with local edits).
    h = hashlib.sha256()
    for p in files:
        h.update(p.relative_to(root).as_posix().encode())
        h.update(p.read_bytes())
    stamp = f"{sha}+src.{h.hexdigest()[:12]}" if sha else f"nogit+src.{h.hexdigest()[:12]}"
    zpath = out_dir / "codmav_src.zip"
    with zipfile.ZipFile(zpath, "w", zipfile.ZIP_DEFLATED) as z:
        for p in files:
            z.write(p, p.relative_to(root).as_posix())
        z.writestr("GIT_SHA", stamp)
    (out_dir / "dataset-metadata.json").write_text(json.dumps(
        {"title": "codmav source", "id": f"{owner}/{slug}", "licenses": [{"name": "other"}]}, indent=2))
    return zpath


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--owner", required=True)
    ap.add_argument("--out", type=Path, default=REPO_ROOT / "dist" / "codmav-src")
    a = ap.parse_args()
    print(build(a.owner, a.out))


if __name__ == "__main__":
    main()
