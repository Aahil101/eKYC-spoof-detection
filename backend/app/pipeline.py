"""Session processing pipeline: decode -> quality -> modules -> fusion/decision -> log.

Milestone 2: modules are placeholders and thresholds are unset, so the decision is
always UNAVAILABLE with an explicit reason. The plumbing (decode, quality vector,
logging, versioning) is real and is what milestone 3/4 plug into.
"""
from __future__ import annotations

import datetime as dt
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import yaml

from backend.app.challenges import challenge_from_json
from backend.app.db import KYCSession, ModuleOutputRow, QualityRow
from backend.app.logging_setup import get_logger
from backend.app.media_store import MediaStore
from ml.common.interface import MODULE_NAMES, KYCModule, ModuleOutput, PreparedSession
from ml.common.media import MediaError, decode_document, decode_video
from ml.common.placeholders import PlaceholderModule
from ml.common.quality import compute_quality

log = get_logger("pipeline")


def load_yaml(path: Path) -> dict:
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def build_modules(bundle_dir: Path | None) -> dict[str, KYCModule]:
    # Milestone 3 replaces this with loaders that read weights from bundle_dir.
    return {m: PlaceholderModule(m) for m in MODULE_NAMES}


def decide(outputs: list[ModuleOutput], policy: dict) -> dict:
    """Decision policy stub honouring the no-fake-numbers rule."""
    ok = [o for o in outputs if o.status == "ok"]
    th = policy.get("thresholds") or {}
    if not ok:
        return {"decision": "UNAVAILABLE", "reason": "No trained modules in the active bundle "
                "(all modules are placeholders). No score was produced.", "fused_score": None,
                "fused_uncertainty": None, "weights": {}, "veto": None}
    if th.get("t_high") is None or th.get("t_low") is None:
        return {"decision": "UNAVAILABLE", "reason": "Decision thresholds have not been calibrated "
                "(config/policy.yaml thresholds are null; run experiment E4).", "fused_score": None,
                "fused_uncertainty": None, "weights": {}, "veto": None}
    raise NotImplementedError("fusion strategies are implemented in milestone 4")


class Pipeline:
    def __init__(self, store: MediaStore, sessionmaker, pipeline_cfg: dict, policy: dict,
                 bundle_version: str, bundle_dir: Path | None, media_ttl_days: int, log_ttl_days: int):
        self.store = store
        self.sm = sessionmaker
        self.cfg = pipeline_cfg
        self.policy = policy
        self.bundle_version = bundle_version
        self.modules = build_modules(bundle_dir)
        self.media_ttl = dt.timedelta(days=media_ttl_days)
        self.log_ttl = dt.timedelta(days=log_ttl_days)

    def process(self, session_id: str) -> None:
        with self.sm() as db:
            s = db.get(KYCSession, session_id)
            if s is None:
                return
            s.status = "processing"
            db.commit()
            challenge = s.challenge_script
        try:
            vcfg, dcfg = self.cfg.get("video", {}), self.cfg.get("document", {})
            doc_meta = {}
            if self.store.exists(session_id, "document"):
                raw = self.store.get(session_id, "document")
                ctype = "application/pdf" if raw[:5] == b"%PDF-" else None
                doc = decode_document(raw, ctype, dcfg.get("max_side", 2000), dcfg.get("pdf_dpi", 200))
            else:
                doc = None
            vid = decode_video(self.store.get(session_id, "video"), vcfg.get("sample_frames", 16),
                               vcfg.get("max_seconds", 12), vcfg.get("max_side", 720))
            q = compute_quality(vid.frames, vid.meta, doc)
            prepared = PreparedSession(session_id=session_id, document_image=doc, frames=vid.frames,
                                       frame_times_ms=vid.times_ms, video_meta=vid.meta,
                                       challenge=challenge_from_json(challenge), quality=q)
            outputs: list[ModuleOutput] = []
            for name, mod in self.modules.items():
                try:
                    outputs.append(mod.predict(prepared))
                except Exception as e:  # a failing module must not kill the session
                    log.error("module_failed", module=name, error=str(e))
                    outputs.append(ModuleOutput(module=name, status="failed", model_version=mod.version,
                                                message=f"{type(e).__name__}: {e}"))
            result = decide(outputs, self.policy)
            self._persist(session_id, outputs, q, vid.meta, doc_meta, result)
            log.info("session_processed", session_id=session_id, decision=result["decision"],
                     bundle_version=self.bundle_version)
        except MediaError as e:
            self._fail(session_id, f"Unreadable upload: {e}")
        except Exception as e:
            log.error("session_failed", session_id=session_id, error=str(e), tb=traceback.format_exc())
            self._fail(session_id, f"Internal error: {type(e).__name__}")

    def _persist(self, session_id, outputs, q, video_meta, doc_meta, result) -> None:
        now = dt.datetime.now(dt.timezone.utc)
        with self.sm() as db:
            s = db.get(KYCSession, session_id)
            if s is None:
                return
            for o in outputs:
                db.add(ModuleOutputRow(session_id=session_id, module=o.module, status=o.status, score=o.score,
                                       raw_score=o.raw_score, confidence=o.confidence, latency_ms=o.latency_ms,
                                       model_version=o.model_version, features=o.features,
                                       explanations=o.explanations, message=o.message))
            # NaN is not valid JSON; store as None
            db.add(QualityRow(session_id=session_id,
                              values={k: (None if v != v else v) for k, v in q.items()}))
            s.video_meta = video_meta
            s.bundle_version = self.bundle_version
            s.decision = result["decision"]
            s.decision_reason = result["reason"]
            s.fused_score = result["fused_score"]
            s.fused_uncertainty = result["fused_uncertainty"]
            s.fusion_strategy = self.policy.get("fusion_strategy")
            s.fusion_weights = result["weights"]
            s.veto_module = result["veto"]
            s.media_expires_at = now + self.media_ttl
            s.log_expires_at = now + self.log_ttl
            s.status = "completed"
            db.commit()

    def _fail(self, session_id: str, msg: str) -> None:
        with self.sm() as db:
            s = db.get(KYCSession, session_id)
            if s:
                s.status, s.error, s.bundle_version = "failed", msg, self.bundle_version
                db.commit()


class JobRunner:
    """In-process worker pool. Interface kept small so Celery/RQ can replace it later."""

    def __init__(self, fn: Callable[[str], None], workers: int = 2, inline: bool = False):
        self.fn, self.inline = fn, inline
        self.pool = None if inline else ThreadPoolExecutor(max_workers=workers, thread_name_prefix="kyc")

    def submit(self, session_id: str) -> None:
        if self.inline:
            self.fn(session_id)
        else:
            self.pool.submit(self.fn, session_id)

    def shutdown(self) -> None:
        if self.pool:
            self.pool.shutdown(wait=False, cancel_futures=True)
