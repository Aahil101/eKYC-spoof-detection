"""OPTIONAL: pull a small, size-capped subset of a registered Kaggle dataset to the laptop.

Default workflow does NOT need this: datasets are attached inside Kaggle notebooks.
Use it only for local debugging, and only within `max_local_mb` from datasets.yaml.

    pip install kaggle            # needs ~/.kaggle/kaggle.json (never commit it)
    python -m training.fetch_subset --dataset ffpp_c23 --max-files 20 --pattern "*.mp4"

Files are listed via the Kaggle API and downloaded one by one until the file or MB cap
is reached, into data/raw/<name>/ (git-ignored).
"""
from __future__ import annotations

import argparse
import fnmatch
from pathlib import Path

import yaml

from training.paths import REPO_ROOT


def _api():
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi
    except ImportError as e:
        raise SystemExit("pip install kaggle  (and put kaggle.json in ~/.kaggle)") from e
    api = KaggleApi()
    api.authenticate()
    return api


def plan(files: list[tuple[str, int]], pattern: str, max_files: int, max_mb: float) -> list[str]:
    chosen, total = [], 0
    for name, size in files:
        if not fnmatch.fnmatch(name, pattern):
            continue
        if len(chosen) >= max_files or (total + size) / 1e6 > max_mb:
            break
        chosen.append(name)
        total += size
    return chosen


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--pattern", default="*")
    ap.add_argument("--max-files", type=int, default=20)
    a = ap.parse_args()
    reg = yaml.safe_load((REPO_ROOT / "training" / "datasets.yaml").read_text(encoding="utf-8"))
    e = reg["datasets"].get(a.dataset)
    if not e or not e.get("kaggle_ref"):
        raise SystemExit(f"{a.dataset}: no verified kaggle_ref in training/datasets.yaml")
    cap = float(e.get("max_local_mb", 0))
    if cap <= 0:
        raise SystemExit(f"{a.dataset}: local download disabled (max_local_mb=0); use it inside Kaggle")
    if e["kind"] == "competition":
        raise SystemExit("competition data: download inside Kaggle only (rules acceptance + size)")
    api = _api()
    listing = api.dataset_list_files(e["kaggle_ref"]).files
    files = [(f.name, int(getattr(f, "totalBytes", 0) or 0)) for f in listing]
    chosen = plan(files, a.pattern, a.max_files, cap)
    out = REPO_ROOT / "data" / "raw" / a.dataset
    out.mkdir(parents=True, exist_ok=True)
    for name in chosen:
        api.dataset_download_file(e["kaggle_ref"], name, path=str(out), quiet=False)
    print(f"downloaded {len(chosen)} files to {out} (cap {cap} MB). Original license: {e['original_license']}")


if __name__ == "__main__":
    main()
