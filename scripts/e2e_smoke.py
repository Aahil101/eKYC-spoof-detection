"""End-to-end smoke test against a running stack (default: docker compose on :8080).

    python scripts/e2e_smoke.py [--base http://127.0.0.1:8080]

Creates a session with synthetic media (no real faces/IDs), uploads, submits, polls,
prints the logged result, then deletes the session (erasure path).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tests.conftest import make_document_png, make_video_bytes  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8080")
    base = ap.parse_args().base.rstrip("/")
    c = httpx.Client(base_url=base, timeout=60)
    cfg = c.get("/api/config/public").raise_for_status().json()
    r = c.post("/api/sessions", json={"consent": True, "consent_version": cfg["consent_version"]}).raise_for_status()
    s = r.json()
    h = {"X-Session-Token": s["token"]}
    sid = s["session_id"]
    c.post(f"/api/sessions/{sid}/document", headers=h,
           files={"file": ("id.png", make_document_png(), "image/png")}).raise_for_status()
    c.post(f"/api/sessions/{sid}/video", headers=h,
           files={"file": ("s.webm", make_video_bytes(n_frames=60, fps=30, size=(320, 240)), "video/webm")}
           ).raise_for_status()
    c.post(f"/api/sessions/{sid}/submit", headers=h).raise_for_status()
    for _ in range(60):
        out = c.get(f"/api/sessions/{sid}", headers=h).raise_for_status().json()
        if out["status"] in ("completed", "failed"):
            break
        time.sleep(0.5)
    summary = {k: out[k] for k in ("status", "decision", "decision_reason", "bundle_version", "placeholder")}
    summary["modules"] = {m["module"]: m["status"] for m in out["modules"]}
    summary["quality_sample"] = {k: out["quality"][k] for k in ("blur_laplacian_var", "fps_measured", "bitrate_bps",
                                                                 "doc_width")} if out["quality"] else None
    print(json.dumps(summary, indent=2))
    c.delete(f"/api/sessions/{sid}", headers=h).raise_for_status()
    gone = c.get(f"/api/sessions/{sid}", headers=h).status_code == 404
    print("erased:", gone)
    return 0 if out["status"] == "completed" and gone else 1


if __name__ == "__main__":
    raise SystemExit(main())
