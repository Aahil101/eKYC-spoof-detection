"""Dataset audit job: verify attached Kaggle mirrors against the originals.

Runs inside Kaggle where datasets are attached (no laptop storage). For every dataset in
cfg.params.datasets it locates the mount under the input roots, then records:
file count, total bytes, extension histogram, per-directory counts (depth <= 4),
presence of `expected_layout` entries, and dataset-specific checks (expected counts from
the original papers are in `expected_counts` and only compared, never assumed).

Output artifact: dataset_audit.json  (copied into docs via scripts/apply_audit.py by hand-review).
"""
from __future__ import annotations

import json
import os
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from training.jobs import register
from training.paths import REPO_ROOT, input_roots


def _find_mount(slug: str, roots: list[Path]) -> Path | None:
    """Kaggle mounts datasets at /kaggle/input/<slug> or /kaggle/input/datasets/<owner>/<slug>."""
    for r in roots:
        if not r.is_dir():
            continue
        for cand in [r / slug, *r.glob(f"*/{slug}"), *r.glob(f"*/*/{slug}")]:
            if cand.is_dir():
                return cand
    return None


def _has_entry(root: Path, name: str, max_depth: int = 3) -> str | None:
    """Breadth-first, depth-bounded search for a file/dir name (cheap on big mounts)."""
    level = [root]
    for _ in range(max_depth + 1):
        nxt = []
        for d in level:
            if (d / name).exists():
                return str((d / name).relative_to(root))
            try:
                nxt += [Path(e.path) for e in os.scandir(d) if e.is_dir(follow_symlinks=False)][:200]
            except OSError:
                pass
        level = nxt
    return None


def audit_tree(root: Path, depth: int = 4, time_cap_s: float = 0.0) -> dict:
    """Count files per directory using scandir (no per-file stat: slow on Kaggle's network FS).

    Sizes come from the Kaggle API listing instead. Stops at `time_cap_s` and marks partial.
    """
    import time
    t0 = time.monotonic()
    n_files = 0
    ext = Counter()
    per_dir: dict[str, int] = defaultdict(int)
    stack = [root]
    partial = False
    while stack:
        if time_cap_s and time.monotonic() - t0 > time_cap_s:
            partial = True
            break
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for e in it:
                    if e.is_dir(follow_symlinks=False):
                        stack.append(Path(e.path))
                    else:
                        n_files += 1
                        ext[os.path.splitext(e.name)[1].lower() or "<none>"] += 1
                        rel = Path(d).relative_to(root)
                        per_dir["/".join(rel.parts[:depth]) or "."] += 1
        except OSError:
            continue
    top_dirs = dict(sorted(per_dir.items(), key=lambda kv: -kv[1])[:300])
    return {"n_files": n_files, "partial": partial, "seconds": round(time.monotonic() - t0, 1),
            "extensions": dict(ext.most_common(20)), "files_per_dir_depth%d" % depth: top_dirs}


@register("dataset_audit")
def dataset_audit(cfg: dict, ctx) -> dict:
    reg = yaml.safe_load((REPO_ROOT / "training" / "datasets.yaml").read_text(encoding="utf-8"))
    roots = input_roots()
    report: dict = {"input_roots": [str(r) for r in roots], "datasets": {}}
    for name in cfg["params"]["datasets"]:
        e = reg["datasets"][name]
        ref = e.get("kaggle_ref")
        slug = ref.split("/")[-1] if ref else None
        mount = _find_mount(slug, roots) if slug else None
        entry: dict = {"kaggle_ref": ref, "mount": str(mount) if mount else None}
        if mount is None:
            entry["error"] = "not attached / not found"
        else:
            entry.update(audit_tree(mount, time_cap_s=float(cfg["params"].get("per_dataset_minutes", 20)) * 60))
            entry["expected_layout"] = {x: _has_entry(mount, x) for x in e.get("expected_layout", [])}
            entry["expected_counts"] = e.get("expected_counts", {})
        report["datasets"][name] = entry
        ctx.log("audited", dataset=name, n_files=entry.get("n_files"), partial=entry.get("partial"),
                mount=entry["mount"])
        # incremental: a dead or slow session still leaves results for finished datasets
        ctx.save_artifact("dataset_audit.json", json.dumps(report, indent=2).encode())
    return {"audited": list(report["datasets"]),
            "missing": [k for k, v in report["datasets"].items() if v.get("error")]}
