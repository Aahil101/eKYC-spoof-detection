"""Integration: trained-head bundle -> API session. Uses synthetic media (no faces), so face modules must
report not_applicable and M1 must score. Needs the pretrained weights (CLIP via HF cache, MediaPipe,
ArcFace in models/store); skipped when they are not available locally."""
import os
import pickle

import numpy as np
import pytest

from ml.registry.bundle import import_module_artifacts

pytest.importorskip("open_clip")
pytest.importorskip("mediapipe")


def _weights_available() -> bool:
    from ml.common.weights import path_for
    try:
        path_for("mediapipe_landmarker", download=False)
        path_for("arcface_buffalo_l", download=False)
        return True
    except FileNotFoundError:
        return False


@pytest.mark.skipif(not _weights_available(), reason="pretrained weights not fetched (python -m ml.common.weights fetch)")
def test_trained_bundle_session(tmp_path, document_png, video_webm, monkeypatch):
    import cv2

    from backend.app.config import Settings
    from backend.app.crypto import new_key_spec
    from backend.app.main import create_app
    from backend.app.schemas import CONSENT_VERSION
    from fastapi.testclient import TestClient
    from ml.features.images import m1_image_features
    from ml.heads import build_matrix, fit_head

    doc = cv2.imdecode(np.frombuffer(document_png, np.uint8), cv2.IMREAD_COLOR)
    s1, v1 = m1_image_features(doc)
    rng = np.random.default_rng(0)
    recs = [{"m1": {k: v * (1 + rng.normal(0, .1)) for k, v in s1.items()},
             "m1_vec": {"spectrum": v1["spectrum"] + rng.normal(0, .05, 24)}} for _ in range(120)]
    X, names, vd, _ = build_matrix(recs, "m1", ("spectrum",))
    art = fit_head("M1", X, rng.integers(0, 2, 120), np.array(["train"] * 80 + ["cal"] * 40), names, vd,
                   ("spectrum",), "m1", kind="hgb", calib="isotonic")
    f = tmp_path / "head_M1.pkl"
    f.write_bytes(pickle.dumps(art))
    import_module_artifacts(tmp_path / "bundles" / "t", "M1", [f], {"note": "test"}, "0.0.1")

    s = Settings(_env_file=None, database_url=f"sqlite:///{(tmp_path / 'db.sqlite').as_posix()}",
                 media_dir=tmp_path / "m", secrets_dir=tmp_path / "s", media_keys=new_key_spec("t"),
                 run_jobs_inline=True, model_bundle="t", bundles_root=tmp_path / "bundles")
    with TestClient(create_app(s)) as c:
        j = c.post("/api/sessions", json={"consent": True, "consent_version": CONSENT_VERSION}).json()
        h = {"X-Session-Token": j["token"]}
        sid = j["session_id"]
        c.post(f"/api/sessions/{sid}/document", headers=h, files={"file": ("id.png", document_png, "image/png")})
        c.post(f"/api/sessions/{sid}/video", headers=h, files={"file": ("v.webm", video_webm, "video/webm")})
        c.post(f"/api/sessions/{sid}/submit", headers=h)
        out = c.get(f"/api/sessions/{sid}", headers=h).json()
    assert out["status"] == "completed", out["error"]
    mods = {m["module"]: m for m in out["modules"]}
    assert mods["M1"]["status"] == "ok" and 0 <= mods["M1"]["score"] <= 1
    assert mods["M1"]["explanations"]["ela_heatmap_png"]
    for m in ("M2", "M3", "M4", "M5"):
        assert mods[m]["status"] in ("placeholder", "not_applicable")   # no head / no face in synthetic video
    assert out["placeholder"] is False and out["bundle_version"].startswith("0.0.1+")
    assert out["decision"] == "UNAVAILABLE"                              # fusion/thresholds: milestone 4
