"""Audit a Kaggle dataset from its file LISTING (metadata API only - nothing is downloaded).

    python scripts/audit_listing.py reubensuju/celeb-df-v2 [--depth 2] [--max-pages 400]

Counts files and bytes per top-level directories and extension. Writes JSON to
docs/audits/<owner>__<slug>.json.
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ref")
    ap.add_argument("--depth", type=int, default=2)
    ap.add_argument("--page-size", type=int, default=1000)
    ap.add_argument("--max-pages", type=int, default=400)
    a = ap.parse_args()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    n = total = pages = 0
    per_dir_n: dict[str, int] = defaultdict(int)
    per_dir_b: dict[str, int] = defaultdict(int)
    ext = Counter()
    token = None
    t0 = time.time()
    while True:
        for attempt in range(8):
            try:
                res = api.dataset_list_files(a.ref, page_token=token, page_size=a.page_size)
                break
            except Exception as e:  # 429 Too Many Requests -> exponential backoff
                if "429" not in str(e) or attempt == 7:
                    raise
                time.sleep(min(60, 5 * 2 ** attempt))
        time.sleep(0.5)
        for f in res.files:
            name = f.name
            size = int(getattr(f, "total_bytes", None) or getattr(f, "totalBytes", 0) or 0)
            parts = name.split("/")
            key = "/".join(parts[: min(a.depth, len(parts) - 1)]) or "."
            per_dir_n[key] += 1
            per_dir_b[key] += size
            ext[Path(name).suffix.lower() or "<none>"] += 1
            n += 1
            total += size
        pages += 1
        token = getattr(res, "next_page_token", None) or getattr(res, "nextPageToken", None)
        if not token or pages >= a.max_pages:
            break
    out = {"ref": a.ref, "complete": not token, "pages": pages, "n_files": n, "total_bytes": total,
           "extensions": dict(ext.most_common(15)),
           "per_dir": {k: {"files": per_dir_n[k], "bytes": per_dir_b[k]} for k in sorted(per_dir_n)},
           "seconds": round(time.time() - t0, 1), "method": "Kaggle API file listing (metadata only)"}
    dest = ROOT / "docs" / "audits" / (a.ref.replace("/", "__") + ".json")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("ref", "complete", "n_files", "total_bytes", "extensions")}))
    for k, v in out["per_dir"].items():
        print(f"  {k:60s} {v['files']:8d} files {v['bytes'] / 1e9:8.2f} GB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
