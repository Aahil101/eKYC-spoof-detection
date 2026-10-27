"""FastAPI application."""
from __future__ import annotations

import datetime as dt
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import select

from backend.app import auth
from backend.app.challenges import challenge_to_json, generate_challenge
from backend.app.config import Settings, get_settings
from backend.app.crypto import KeyRing
from backend.app.db import KYCSession, Label, init_db, make_engine, make_sessionmaker
from backend.app.logging_setup import configure_logging, get_logger
from backend.app.media_store import MediaStore
from backend.app.pipeline import JobRunner, Pipeline, load_yaml
from backend.app.retention import purge_expired
from backend.app.schemas import (CONSENT_VERSION, ChallengeStepOut, CreateSessionRequest, CreateSessionResponse,
                                 LabelIn, ModuleOut, PublicConfig, SessionOut)
from ml.common.interface import MODULE_TITLES
from ml.registry.bundle import PLACEHOLDER_BUNDLE_VERSION, resolve_active_bundle

log = get_logger("api")

DOC_MAGIC = {b"\xff\xd8\xff": "image/jpeg", b"\x89PNG": "image/png", b"%PDF-": "application/pdf",
             b"RIFF": "image/webp"}
VIDEO_MAGIC = (b"\x1a\x45\xdf\xa3",)  # WebM/Matroska EBML header; MP4/MOV checked via 'ftyp'


def _sniff_document(data: bytes) -> str | None:
    for magic, ctype in DOC_MAGIC.items():
        if data.startswith(magic):
            if ctype == "image/webp" and data[8:12] != b"WEBP":
                return None
            return ctype
    return None


def _sniff_video(data: bytes) -> bool:
    return data.startswith(VIDEO_MAGIC) or data[4:8] == b"ftyp"


async def _read_limited(upload: UploadFile, limit: int) -> bytes:
    buf = bytearray()
    while chunk := await upload.read(1 << 20):
        buf.extend(chunk)
        if len(buf) > limit:
            raise HTTPException(status.HTTP_413_CONTENT_TOO_LARGE, f"file exceeds {limit} bytes")
    if not buf:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "empty file")
    return bytes(buf)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    settings.validate_security()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        app.state.runner.shutdown()

    app = FastAPI(title="codmav Video-KYC API", version="0.2.0", lifespan=lifespan)
    app.state.settings = settings
    app.add_middleware(CORSMiddleware, allow_origins=[o for o in settings.cors_origins.split(",") if o],
                       allow_methods=["GET", "POST", "DELETE"], allow_headers=["*"])

    engine = make_engine(settings.database_url)
    if settings.database_url.startswith("sqlite:///"):
        from pathlib import Path
        Path(settings.database_url.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    init_db(engine)
    sm = make_sessionmaker(engine)
    keyring = KeyRing.load_or_create(settings.media_keys, settings.media_active_key_id, settings.secrets_dir,
                                     allow_create=settings.demo_mode)
    store = MediaStore(settings.media_dir, keyring)
    bundle_version, bundle_dir = resolve_active_bundle(settings.bundles_root, settings.model_bundle)
    pipeline_cfg = load_yaml(settings.config_dir / "pipeline.yaml")
    policy = load_yaml(settings.config_dir / "policy.yaml")
    pipeline = Pipeline(store, sm, pipeline_cfg, policy, bundle_version, bundle_dir,
                        settings.media_ttl_days, settings.log_ttl_days)
    app.state.sm, app.state.store, app.state.pipeline = sm, store, pipeline
    app.state.runner = JobRunner(pipeline.process, inline=settings.run_jobs_inline)
    app.state.bundle_version = bundle_version
    log.info("startup", demo_mode=settings.demo_mode, bundle_version=bundle_version,
             db=settings.database_url.split("://")[0])

    def get_session_authorized(session_id: str, request: Request,
                               x_session_token: str | None = Header(default=None)) -> KYCSession:
        with sm() as db:
            s = db.get(KYCSession, session_id)
        if s is None or s.status == "deleted":
            raise HTTPException(status.HTTP_404_NOT_FOUND, "session not found")
        if not (auth.token_matches(x_session_token, s.token_hash) or auth.is_admin(request)):
            raise HTTPException(status.HTTP_403_FORBIDDEN, "invalid session token")
        return s

    @app.get("/api/health")
    def health():
        return {"status": "ok", "bundle_version": bundle_version}

    @app.get("/api/config/public", response_model=PublicConfig)
    def public_config():
        return PublicConfig(demo_mode=settings.demo_mode, consent_version=CONSENT_VERSION,
                            media_ttl_days=settings.media_ttl_days, bundle_version=bundle_version,
                            placeholder_bundle=bundle_version == PLACEHOLDER_BUNDLE_VERSION,
                            max_document_bytes=settings.max_document_bytes,
                            max_video_bytes=settings.max_video_bytes)

    @app.post("/api/sessions", response_model=CreateSessionResponse, status_code=201)
    def create_session(body: CreateSessionRequest):
        if not body.consent:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "explicit consent is required")
        if body.consent_version != CONSENT_VERSION:
            raise HTTPException(status.HTTP_409_CONFLICT, "consent text has changed; reload and re-consent")
        ccfg = pipeline_cfg.get("challenge", {})
        duration = int(ccfg.get("duration_ms", 7000))
        steps = generate_challenge(duration, int(ccfg.get("steps", 3)), int(ccfg.get("digits_len", 3)))
        token, token_hash = auth.new_token()
        meta = {k: str(v)[:200] for k, v in list(body.client_meta.items())[:20]}
        with sm() as db:
            s = KYCSession(token_hash=token_hash, consent_version=body.consent_version,
                           challenge_script=challenge_to_json(steps), client_meta=meta,
                           bundle_version=bundle_version,
                           media_expires_at=dt.datetime.now(dt.timezone.utc) + dt.timedelta(days=1))
            db.add(s)
            db.commit()
            sid = s.id
        return CreateSessionResponse(session_id=sid, token=token, duration_ms=duration,
                                     challenge=[ChallengeStepOut(kind=c.kind, prompt=c.prompt, start_ms=c.start_ms,
                                                                 end_ms=c.end_ms) for c in steps])

    def _check_open(s: KYCSession):
        if s.status not in ("created", "uploading"):
            raise HTTPException(status.HTTP_409_CONFLICT, f"session is {s.status}")

    @app.post("/api/sessions/{session_id}/document", status_code=204)
    async def upload_document(file: UploadFile = File(...), s: KYCSession = Depends(get_session_authorized)):
        _check_open(s)
        data = await _read_limited(file, settings.max_document_bytes)
        if _sniff_document(data) is None:
            raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "document must be JPEG, PNG, WebP or PDF")
        store.put(s.id, "document", data)
        _mark_uploading(s.id)

    @app.post("/api/sessions/{session_id}/video", status_code=204)
    async def upload_video(file: UploadFile = File(...), s: KYCSession = Depends(get_session_authorized)):
        _check_open(s)
        data = await _read_limited(file, settings.max_video_bytes)
        if not _sniff_video(data):
            raise HTTPException(status.HTTP_415_UNSUPPORTED_MEDIA_TYPE, "video must be WebM or MP4")
        store.put(s.id, "video", data)
        _mark_uploading(s.id)

    def _mark_uploading(sid: str):
        with sm() as db:
            row = db.get(KYCSession, sid)
            row.status = "uploading"
            db.commit()

    @app.post("/api/sessions/{session_id}/submit", status_code=202)
    def submit(s: KYCSession = Depends(get_session_authorized)):
        _check_open(s)
        if not store.exists(s.id, "video"):
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "selfie video not uploaded")
        if not store.exists(s.id, "document"):
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "ID document not uploaded")
        with sm() as db:
            row = db.get(KYCSession, s.id)
            row.status = "queued"
            db.commit()
        app.state.runner.submit(s.id)
        return {"session_id": s.id, "status": "queued"}

    @app.get("/api/sessions/{session_id}", response_model=SessionOut)
    def get_session(s: KYCSession = Depends(get_session_authorized)):
        with sm() as db:
            row = db.get(KYCSession, s.id)
            mods = sorted(row.modules, key=lambda m: m.module)
            q = row.quality.values if row.quality else None
            return SessionOut(
                session_id=row.id, status=row.status, created_at=row.created_at,
                bundle_version=row.bundle_version, decision=row.decision, decision_reason=row.decision_reason,
                fused_score=row.fused_score, fused_uncertainty=row.fused_uncertainty,
                fusion_strategy=row.fusion_strategy, fusion_weights=row.fusion_weights, veto_module=row.veto_module,
                error=row.error, video_meta=row.video_meta, quality=q,
                challenge=[ChallengeStepOut(**{k: c[k] for k in ("kind", "prompt", "start_ms", "end_ms")})
                           for c in row.challenge_script],
                modules=[ModuleOut(module=m.module, title=MODULE_TITLES[m.module], status=m.status, score=m.score,
                                   confidence=m.confidence, model_version=m.model_version, latency_ms=m.latency_ms,
                                   message=m.message, features=m.features, explanations=m.explanations)
                         for m in mods],
                placeholder=(row.bundle_version == PLACEHOLDER_BUNDLE_VERSION),
            )

    @app.delete("/api/sessions/{session_id}", status_code=204)
    def delete_session(s: KYCSession = Depends(get_session_authorized)):
        """Right to erasure: removes media, templates, features, scores and labels."""
        store.delete_session(s.id)
        with sm() as db:
            row = db.get(KYCSession, s.id)
            db.delete(row)
            db.commit()
        log.info("session_erased", session_id=s.id)

    # ---------------- admin / monitor ----------------
    @app.get("/api/admin/sessions", dependencies=[Depends(auth.require_admin)])
    def list_sessions(limit: int = 50, offset: int = 0):
        limit = max(1, min(limit, 500))
        with sm() as db:
            rows = db.scalars(select(KYCSession).order_by(KYCSession.created_at.desc())
                              .limit(limit).offset(max(offset, 0))).all()
            return [{"session_id": r.id, "created_at": r.created_at, "status": r.status, "decision": r.decision,
                     "fused_score": r.fused_score, "bundle_version": r.bundle_version, "source": r.source}
                    for r in rows]

    @app.post("/api/admin/sessions/{session_id}/label", status_code=201, dependencies=[Depends(auth.require_admin)])
    def add_label(session_id: str, body: LabelIn):
        with sm() as db:
            if db.get(KYCSession, session_id) is None:
                raise HTTPException(status.HTTP_404_NOT_FOUND, "session not found")
            db.add(Label(session_id=session_id, label=body.label, attack_type=body.attack_type,
                         source="human_review"))
            db.commit()
        return {"ok": True}

    @app.post("/api/admin/purge", dependencies=[Depends(auth.require_admin)])
    def purge():
        return purge_expired(sm, store)

    # ---------------- Dashboard 2: monitoring (admin) ----------------
    from backend.app.db import Alert, DriftReport
    from monitoring.service import load_reference, run_window, simulate
    app.state.reference = load_reference(bundle_dir)

    def _ref():
        if app.state.reference is None:
            raise HTTPException(status.HTTP_409_CONFLICT, "active bundle has no MONITOR/reference_served.pkl")
        return app.state.reference

    @app.get("/api/monitor/status", dependencies=[Depends(auth.require_admin)])
    def monitor_status():
        r = app.state.reference
        with sm() as db:
            n = db.query(KYCSession).filter(KYCSession.status == "completed",
                                            KYCSession.bundle_version == bundle_version).count()
            n_alerts = db.query(Alert).filter(Alert.status == "open").count()
        return {"bundle_version": bundle_version, "reference_available": r is not None,
                "reference_n": int(len(r["reference"]["Q"])) if r else 0, "window": r.get("window") if r else None,
                "quality_names": r["quality_names"] if r else [], "logged_sessions": n, "open_alerts": n_alerts,
                "params": r.get("monitor") if r else None,
                "novelty_space": (r["novelty"].meta if r else None)}

    @app.post("/api/monitor/run", dependencies=[Depends(auth.require_admin)])
    def monitor_run(window: int | None = None, alpha: float | None = None, alpha_novel: float | None = None,
                    source: str | None = None):
        params = {k: v for k, v in (("alpha", alpha), ("alpha_novel", alpha_novel)) if v is not None}
        return run_window(sm, _ref(), bundle_version, window, params, source)

    @app.post("/api/monitor/simulate", dependencies=[Depends(auth.require_admin)])
    def monitor_simulate(scenario: str = "stationary", n: int = 200, seed: int = 0):
        if scenario not in ("stationary", "domain_drift", "unseen_generator"):
            raise HTTPException(status.HTTP_400_BAD_REQUEST, "scenario must be stationary|domain_drift|unseen_generator")
        return simulate(sm, _ref(), bundle_version, scenario, max(10, min(n, 2000)), seed)

    @app.get("/api/monitor/reports", dependencies=[Depends(auth.require_admin)])
    def monitor_reports(limit: int = 100):
        with sm() as db:
            rows = db.scalars(select(DriftReport).order_by(DriftReport.created_at.desc()).limit(max(1, min(limit, 500))))
            return [{"id": r.id, "created_at": r.created_at, "window_start": r.window_start, "window_end": r.window_end,
                     "label": r.label, "bundle_version": r.bundle_version,
                     "confidence": r.multivariate.get("confidence"), "tests": r.multivariate.get("tests"),
                     "features": [{k: f.get(k) for k in ("feature", "ks_q", "psi", "psi_crit", "drifted", "mean_shift_sd")}
                                  for f in r.per_feature.get("features", [])]} for r in rows][::-1]

    @app.get("/api/monitor/reports/{report_id}", dependencies=[Depends(auth.require_admin)])
    def monitor_report(report_id: int):
        with sm() as db:
            r = db.get(DriftReport, report_id)
            if r is None:
                raise HTTPException(status.HTTP_404_NOT_FOUND, "report not found")
            return {"id": r.id, "label": r.label, "window_start": r.window_start, "window_end": r.window_end,
                    "bundle_version": r.bundle_version, "per_feature": r.per_feature, "multivariate": r.multivariate,
                    "streaming": r.streaming}

    @app.get("/api/monitor/alerts", dependencies=[Depends(auth.require_admin)])
    def monitor_alerts(status_filter: str | None = None):
        with sm() as db:
            q = select(Alert).order_by(Alert.created_at.desc())
            if status_filter:
                q = q.where(Alert.status == status_filter)
            return [{"id": a.id, "created_at": a.created_at, "type": a.type, "severity": a.severity,
                     "confidence": a.confidence, "bundle_version": a.bundle_version, "status": a.status,
                     "recommendation": a.recommendation, "evidence": a.evidence} for a in db.scalars(q.limit(200))]

    @app.post("/api/monitor/alerts/{alert_id}/ack", dependencies=[Depends(auth.require_admin)])
    def monitor_ack(alert_id: int):
        with sm() as db:
            a = db.get(Alert, alert_id)
            if a is None:
                raise HTTPException(status.HTTP_404_NOT_FOUND, "alert not found")
            a.status, a.acknowledged_by = "acknowledged", "admin"
            db.commit()
        return {"ok": True}

    return app


app = None


def get_app() -> FastAPI:  # uvicorn factory: uvicorn backend.app.main:get_app --factory
    return create_app()
