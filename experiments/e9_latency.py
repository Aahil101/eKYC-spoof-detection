"""E9 latency: end-to-end session processing time and per-module latency on the current machine (CPU).

    python -m experiments.e9_latency --videos a.mp4 b.mp4 [--repeats 3] [--portrait face.png]

Runs real sessions through the FastAPI app (TestClient, inline jobs, temporary SQLite/media dirs) with the bundle
in MODEL_BUNDLE. The document is the synthetic test document (tests/conftest.py); a portrait image may be pasted
in so that M5 has a face. Media never leave the temp directory. Reports per-module latency_ms as stored in
module_results, plus wall time of /submit.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import tempfile
import time
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", nargs="+", required=True)
    ap.add_argument("--portrait", default=None)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("experiments/results/e9"))
    a = ap.parse_args()
    tmp = Path(tempfile.mkdtemp(prefix="codmav_e9_"))
    os.environ.setdefault("MODEL_BUNDLE", "kyc-v0.4")
    os.environ.update(RUN_JOBS_INLINE="1", DATABASE_URL=f"sqlite:///{(tmp / 'e9.db').as_posix()}",
                      MEDIA_DIR=str(tmp / "media"), SECRETS_DIR=str(tmp / "secrets"))
    import cv2
    from fastapi.testclient import TestClient
    from backend.app.main import create_app
    from backend.app.schemas import CONSENT_VERSION
    from tests.conftest import make_document_png
    c = TestClient(create_app())
    doc = cv2.imdecode(np.frombuffer(make_document_png(800, 500), np.uint8), cv2.IMREAD_COLOR)
    if a.portrait:
        face = cv2.resize(cv2.imread(a.portrait), (224, 224))
        doc[100:324, 40:264] = face
    doc_png = cv2.imencode(".png", doc)[1].tobytes()
    rows = []
    for v in a.videos:
        vid = open(v, "rb").read()
        for r in range(a.repeats + 1):                       # first run = warm-up (model loading), reported separately
            s = c.post("/api/sessions", json={"consent": True, "consent_version": CONSENT_VERSION}).json()
            h = {"X-Session-Token": s["token"]}
            c.post(f"/api/sessions/{s['session_id']}/document", headers=h, files={"file": ("id.png", doc_png, "image/png")})
            c.post(f"/api/sessions/{s['session_id']}/video", headers=h, files={"file": ("v.mp4", vid, "video/mp4")})
            t = time.perf_counter()
            c.post(f"/api/sessions/{s['session_id']}/submit", headers=h)
            wall = time.perf_counter() - t
            o = c.get(f"/api/sessions/{s['session_id']}", headers=h).json()
            rows.append({"video": Path(v).name, "warmup": r == 0, "wall_s": wall, "status": o["status"],
                         "modules": {m["module"]: m.get("latency_ms") for m in o["modules"]}})
            print(rows[-1], flush=True)
    steady = [x for x in rows if not x["warmup"]]
    mods = sorted({m for x in steady for m in x["modules"]})
    summ = {"machine": {"platform": platform.platform(), "processor": platform.processor(), "cpu_count": os.cpu_count()},
            "bundle": os.environ["MODEL_BUNDLE"], "n_sessions": len(steady),
            "wall_s": {"median": float(np.median([x["wall_s"] for x in steady])), "max": float(np.max([x["wall_s"] for x in steady]))},
            "warmup_wall_s": [x["wall_s"] for x in rows if x["warmup"]],
            "module_latency_ms_median": {m: float(np.median([x["modules"][m] for x in steady if x["modules"].get(m) is not None]))
                                         for m in mods if any(x["modules"].get(m) is not None for x in steady)}}
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "e9.json").write_text(json.dumps({"summary": summ, "rows": rows}, indent=2))
    L = [f"## E9: latency ({summ['machine']['processor'] or summ['machine']['platform']}, {summ['machine']['cpu_count']} logical CPUs, no GPU)", "",
         f"Sessions: {summ['n_sessions']} steady-state (after 1 warm-up per video). End-to-end /submit wall time: median "
         f"{summ['wall_s']['median']:.1f} s, max {summ['wall_s']['max']:.1f} s; warm-up {', '.join(f'{w:.1f}' for w in summ['warmup_wall_s'])} s.", "",
         "Stored module latency = the module's own scoring time + 1/5 of the SHARED media analysis (decode, face",
         "tracking, CLIP/LBP/landmark features; ml/modules.py), so the near-equal ~2 s values are mostly the shared",
         "analysis; the excess of a module over the minimum is its own cost.", "",
         "| module | median latency (ms) |", "|---|---|"]
    L += [f"| {m} | {v:.0f} |" for m, v in summ["module_latency_ms_median"].items()]
    (a.out / "e9.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    main()
