"""Fetch a public archive from its ORIGINAL host inside a Kaggle notebook (internet on).

Used for datasets without a Kaggle mirror whose license allows re-hosting (e.g. SIDTD,
CC-BY-4.0). The extracted files land in /kaggle/working/data/<name>/ and become this
notebook's output; other notebooks attach it via kernel-metadata `kernel_sources`
(datasets.yaml kind: kernel_output). Nothing is downloaded to the laptop.

The artifact `fetch_manifest.json` records URL, byte size, sha256 of the archive,
HTTP headers and extracted file counts, so the copy is auditable against the original.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

from training.jobs import register
from training.paths import KAGGLE_WORKING, REPO_ROOT, is_kaggle


@register("fetch_archive")
def fetch_archive(cfg: dict, ctx) -> dict:
    p = cfg["params"]
    base = KAGGLE_WORKING if is_kaggle() else REPO_ROOT / "runs"
    dest = Path(os.environ.get("CODMAV_FETCH_DEST", base / "data")) / p["name"]
    dest.mkdir(parents=True, exist_ok=True)
    tmp = dest.parent / f".{p['name']}.download"
    h = hashlib.sha256()
    req = urllib.request.Request(p["url"], headers={"User-Agent": "codmav-research-fetch/1.0"})
    max_bytes = int(p.get("max_bytes") or 0)
    n = 0
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        headers = {k: v for k, v in r.headers.items() if k.lower() in ("content-length", "last-modified", "etag")}
        while chunk := r.read(1 << 22):
            f.write(chunk)
            h.update(chunk)
            n += len(chunk)
            if max_bytes and n >= max_bytes:
                ctx.log("truncated_by_max_bytes", bytes=n)
                break
            if n % (256 << 20) < (1 << 22):
                ctx.log("progress", mb=round(n / 2**20))
    digest = h.hexdigest()
    if p.get("sha256") and p["sha256"] != digest:
        raise RuntimeError(f"sha256 mismatch: {digest} != {p['sha256']}")
    extracted, ext = 0, Counter()
    if p.get("extract", True) and not (max_bytes and n >= max_bytes):
        with zipfile.ZipFile(tmp) as z:
            for m in z.infolist():
                target = (dest / m.filename).resolve()
                if not str(target).startswith(str(dest.resolve())):   # zip-slip guard
                    raise RuntimeError(f"unsafe path in archive: {m.filename}")
                z.extract(m, dest)
                if not m.is_dir():
                    extracted += 1
                    ext[Path(m.filename).suffix.lower()] += 1
        tmp.unlink()
    else:
        shutil.move(tmp, dest / Path(p["url"]).name)
    top = sorted(x.name for x in dest.iterdir())[:50]
    manifest = {"name": p["name"], "url": p["url"], "license": p.get("license"), "bytes": n, "sha256": digest,
                "headers": headers, "extracted_files": extracted, "extensions": dict(ext.most_common(15)),
                "top_level": top, "dest": str(dest)}
    ctx.save_artifact("fetch_manifest.json", json.dumps(manifest, indent=2).encode())
    (dest / "FETCH_MANIFEST.json").write_text(json.dumps(manifest, indent=2))
    if p.get("license_notice"):
        (dest / "LICENSE_NOTICE.txt").write_text(p["license_notice"])
    return {"bytes": n, "sha256": digest, "extracted_files": extracted}
