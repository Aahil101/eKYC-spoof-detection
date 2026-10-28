import datetime as dt

import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.crypto import new_key_spec
from backend.app.db import KYCSession
from backend.app.main import create_app
from backend.app.retention import purge_expired
from backend.app.schemas import CONSENT_VERSION
from ml.registry.bundle import PLACEHOLDER_BUNDLE_VERSION


def make_client(tmp_path, **over) -> TestClient:
    s = Settings(_env_file=None, database_url=f"sqlite:///{(tmp_path / 't.db').as_posix()}",
                 media_dir=tmp_path / "media", secrets_dir=tmp_path / "secrets", media_keys=new_key_spec("t"),
                 run_jobs_inline=True, **over)
    return TestClient(create_app(s))


@pytest.fixture
def client(tmp_path):
    with make_client(tmp_path) as c:
        yield c


def new_session(c):
    r = c.post("/api/sessions", json={"consent": True, "consent_version": CONSENT_VERSION})
    assert r.status_code == 201, r.text
    j = r.json()
    return j["session_id"], {"X-Session-Token": j["token"]}, j


def test_health_and_public_config(client):
    assert client.get("/api/health").json()["bundle_version"] == PLACEHOLDER_BUNDLE_VERSION
    cfg = client.get("/api/config/public").json()
    assert cfg["placeholder_bundle"] is True and cfg["demo_mode"] is True


def test_consent_required(client):
    assert client.post("/api/sessions", json={"consent": False, "consent_version": CONSENT_VERSION}).status_code == 400
    assert client.post("/api/sessions", json={"consent": True, "consent_version": "old"}).status_code == 409


def test_full_flow_logs_quality_and_placeholder_outputs(client, video_webm, document_png):
    sid, h, created = new_session(client)
    # default config = free (self-paced) recording: no timed prompts, actions are detected afterwards
    assert created["challenge"] == [] and created["duration_ms"] == 30000
    assert client.post(f"/api/sessions/{sid}/document", headers=h,
                       files={"file": ("id.png", document_png, "image/png")}).status_code == 204
    assert client.post(f"/api/sessions/{sid}/video", headers=h,
                       files={"file": ("s.webm", video_webm, "video/webm")}).status_code == 204
    assert client.post(f"/api/sessions/{sid}/submit", headers=h).status_code == 202
    s = client.get(f"/api/sessions/{sid}", headers=h).json()
    assert s["status"] == "completed", s
    assert s["decision"] == "UNAVAILABLE"                # no trained models -> no decision
    assert s["fused_score"] is None
    assert s["placeholder"] is True and s["bundle_version"] == PLACEHOLDER_BUNDLE_VERSION
    assert {m["module"] for m in s["modules"]} == {"M1", "M2", "M3", "M4", "M5"}
    assert all(m["status"] == "placeholder" and m["score"] is None for m in s["modules"])
    q = s["quality"]
    assert q["blur_laplacian_var"] > 0 and q["doc_width"] == 400 and q["face_height_frac"] is None
    # resubmission is blocked
    assert client.post(f"/api/sessions/{sid}/submit", headers=h).status_code == 409


def test_token_required_and_isolated(client, document_png):
    sid, h, _ = new_session(client)
    sid2, h2, _ = new_session(client)
    files = {"file": ("id.png", document_png, "image/png")}
    assert client.post(f"/api/sessions/{sid}/document", files=files).status_code in (403, 422)
    # TestClient is "testclient" host (not loopback) so admin bypass does not apply
    assert client.post(f"/api/sessions/{sid}/document", headers=h2, files=files).status_code == 403
    assert client.get(f"/api/sessions/{sid2}", headers=h).status_code == 403


def test_upload_validation(client, video_webm):
    sid, h, _ = new_session(client)
    r = client.post(f"/api/sessions/{sid}/document", headers=h, files={"file": ("x.png", b"MZ\x90\x00exe", "image/png")})
    assert r.status_code == 415
    r = client.post(f"/api/sessions/{sid}/video", headers=h, files={"file": ("x.webm", b"hello", "video/webm")})
    assert r.status_code == 415
    assert client.post(f"/api/sessions/{sid}/submit", headers=h).status_code == 400


def test_upload_size_limit(tmp_path, video_webm):
    with make_client(tmp_path, max_video_bytes=1000) as c:
        sid, h, _ = new_session(c)
        r = c.post(f"/api/sessions/{sid}/video", headers=h, files={"file": ("s.webm", video_webm, "video/webm")})
        assert r.status_code == 413


def test_corrupt_video_marks_session_failed(client, document_png):
    sid, h, _ = new_session(client)
    client.post(f"/api/sessions/{sid}/document", headers=h, files={"file": ("id.png", document_png, "image/png")})
    client.post(f"/api/sessions/{sid}/video", headers=h,
                files={"file": ("s.webm", b"\x1a\x45\xdf\xa3" + b"\x00" * 500, "video/webm")})
    client.post(f"/api/sessions/{sid}/submit", headers=h)
    s = client.get(f"/api/sessions/{sid}", headers=h).json()
    assert s["status"] == "failed" and "Unreadable" in s["error"]


def test_delete_erases_everything(client, tmp_path, video_webm, document_png):
    sid, h, _ = new_session(client)
    client.post(f"/api/sessions/{sid}/document", headers=h, files={"file": ("id.png", document_png, "image/png")})
    assert (tmp_path / "media" / sid / "document.bin").exists()
    assert client.delete(f"/api/sessions/{sid}", headers=h).status_code == 204
    assert not (tmp_path / "media" / sid).exists()
    assert client.get(f"/api/sessions/{sid}", headers=h).status_code == 404


def test_admin_requires_key_when_configured(tmp_path):
    with make_client(tmp_path, admin_api_key="s3cret") as c:
        assert c.get("/api/admin/sessions").status_code == 401
        assert c.get("/api/admin/sessions", headers={"X-API-Key": "wrong"}).status_code == 401
        assert c.get("/api/admin/sessions", headers={"X-API-Key": "s3cret"}).status_code == 200


def test_admin_blocked_for_non_loopback_in_demo(client):
    # TestClient's client host is "testclient", i.e. not loopback
    assert client.get("/api/admin/sessions").status_code == 401


def test_non_demo_requires_admin_key(tmp_path):
    with pytest.raises(RuntimeError):
        make_client(tmp_path, demo_mode=False)


def test_label_and_purge(tmp_path, video_webm, document_png):
    with make_client(tmp_path, admin_api_key="k") as c:
        sid, h, _ = new_session(c)
        c.post(f"/api/sessions/{sid}/document", headers=h, files={"file": ("id.png", document_png, "image/png")})
        c.post(f"/api/sessions/{sid}/video", headers=h, files={"file": ("s.webm", video_webm, "video/webm")})
        c.post(f"/api/sessions/{sid}/submit", headers=h)
        ak = {"X-API-Key": "k"}
        assert c.post(f"/api/admin/sessions/{sid}/label", headers=ak,
                      json={"label": "genuine"}).status_code == 201
        assert c.post(f"/api/admin/sessions/{sid}/label", headers=ak, json={"label": "maybe"}).status_code == 422
        # media expires: purge removes media but keeps the log row
        sm = c.app.state.sm
        with sm() as db:
            row = db.get(KYCSession, sid)
            row.media_expires_at = dt.datetime.now(dt.timezone.utc) - dt.timedelta(seconds=1)
            db.commit()
        assert purge_expired(sm, c.app.state.store)["media_purged"] == 1
        assert not (tmp_path / "media" / sid).exists()
        assert c.get(f"/api/sessions/{sid}", headers=h).json()["status"] == "completed"
