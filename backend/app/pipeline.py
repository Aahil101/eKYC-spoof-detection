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
    """Trained heads from the bundle where present; placeholders otherwise."""
    if bundle_dir is None:
        return {m: PlaceholderModule(m) for m in MODULE_NAMES}
    from ml.modules import load_modules
    return load_modules(bundle_dir)


def _jsonable(x):
    import numpy as np
    if isinstance(x, dict):
        return {str(k): _jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple, set)):
        return [_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return _jsonable(x.tolist())
    if isinstance(x, np.str_):
        return str(x)
    if isinstance(x, (np.floating, float)):
        return None if x != x or x in (float("inf"), float("-inf")) else float(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    return x


def load_fusion(bundle_dir: Path | None):
    """FUSION/fusion.pkl from the (hash-verified) bundle, or None."""
    if bundle_dir is None:
        return None
    f = bundle_dir / "FUSION" / "fusion.pkl"
    if not f.exists():
        return None
    import pickle
    with open(f, "rb") as fh:
        return pickle.load(fh)


def decide(outputs: list[ModuleOutput], policy: dict, fusion: dict | None = None,
           quality: dict | None = None) -> dict:
    """Fusion + data-driven policy from the bundle; UNAVAILABLE when either is missing (no guessed numbers)."""
    import numpy as np
    ok = [o for o in outputs if o.status == "ok"]
    if not ok:
        return {"decision": "UNAVAILABLE", "reason": "No trained modules in the active bundle "
                "(all modules are placeholders). No score was produced.", "fused_score": None,
                "fused_uncertainty": None, "weights": {}, "veto": None}
    m4 = next((o for o in ok if o.module == "M4"), None)
    challenge_failed = bool(m4 and m4.explanations.get("challenge_failed"))
    if fusion is None:
        return {"decision": "UNAVAILABLE", "reason": "Per-module scores are available, but the bundle has no "
                "calibrated fusion/decision policy (FUSION/fusion.pkl)."
                + (" Liveness challenge not completed." if challenge_failed else ""),
                "fused_score": None, "fused_uncertainty": None, "weights": {}, "veto": None}
    from ml.fusion import MODULES, FusionInput, quality_matrix
    by = {o.module: o for o in ok}
    P = np.array([[by[m].score if m in by else np.nan for m in MODULES]], float)
    C = np.array([[by[m].confidence if m in by else np.nan for m in MODULES]], float)
    fi = FusionInput(P, C, quality_matrix([quality or {}]))
    p, w, u = fusion["model"].predict(fi)
    pol = fusion["policy"]
    Pd = {m: (by[m].score if m in by else None) for m in MODULES}
    Cd = {m: (by[m].confidence if m in by else None) for m in MODULES}
    decision, veto, reason = pol.decide(float(p[0]), Pd, Cd, challenge_failed)
    # self-paced liveness actions (free recording mode): detection thresholds are heuristic and unvalidated,
    # so missing actions cap the outcome at HUMAN_REVIEW instead of forcing a REJECT
    acts = (m4.explanations.get("actions") if m4 else None) or {}
    if acts and not acts.get("all_done") and decision == "ACCEPT":
        missing = [x["label"].lower() for x in acts.get("actions", []) if not x["done"]]
        decision, reason = "HUMAN_REVIEW", ("The checks passed, but these liveness actions were not detected: "
                                            + ", ".join(missing) + ". A reviewer will look at this session.")
    if decision == "REJECT" and veto is None:
        worst = min(ok, key=lambda o: o.score)
        reason = f"{reason} Lowest check: {worst.module} ({_REASON.get(worst.module, worst.module)})."
    weights = {m: (None if not np.isfinite(w[0, j]) else round(float(w[0, j]), 4)) for j, m in enumerate(MODULES)}
    return {"decision": decision, "reason": reason, "fused_score": float(p[0]),
            "fused_uncertainty": None if not np.isfinite(u[0]) else float(u[0]),
            "weights": {"strategy": fusion["strategy"], "per_module": weights,
                        "thresholds": {"t_high": pol.t_high, "t_low": pol.t_low}, "veto": pol.veto},
            "veto": veto}


_REASON = {"M1": "the ID document looks altered or recaptured", "M2": "the video may be a deepfake",
           "M3": "the video may be a replay from a screen or photo", "M4": "liveness could not be confirmed",
           "M5": "the face does not match the ID photo"}


def _legacy_decide(outputs: list[ModuleOutput], policy: dict) -> dict:
    """Decision policy stub honouring the no-fake-numbers rule (fusion arrives in milestone 4)."""
    ok = [o for o in outputs if o.status == "ok"]
    th = policy.get("thresholds") or {}
    if not ok:
        return {"decision": "UNAVAILABLE", "reason": "No trained modules in the active bundle "
                "(all modules are placeholders). No score was produced.", "fused_score": None,
                "fused_uncertainty": None, "weights": {}, "veto": None}
    if th.get("t_high") is None or th.get("t_low") is None:
        notes = []
        m4 = next((o for o in ok if o.module == "M4"), None)
        if m4 and m4.explanations.get("challenge_failed"):
            notes.append("Liveness challenge not completed (hard rule: would force REJECT/retry).")
        return {"decision": "UNAVAILABLE", "reason": "Per-module scores are available, but fusion and "
                "decision thresholds are not calibrated yet (milestone 4, experiment E4). "
                + " ".join(notes), "fused_score": None, "fused_uncertainty": None, "weights": {}, "veto": None}
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
        self.fusion = load_fusion(bundle_dir)
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
            vid_bytes = self.store.get(session_id, "video")
            vid = decode_video(vid_bytes, vcfg.get("sample_frames", 16),
                               vcfg.get("max_seconds", 12), vcfg.get("max_side", 720),
                               full_every_ms=float(vcfg.get("full_every_ms", 0)))
            steps = challenge_from_json(challenge)
            trained = any(not isinstance(m, PlaceholderModule) for m in self.modules.values())
            analysis = None
            if trained:
                from ml.modules import analyse_session
                analysis = analyse_session(vid_bytes, doc, steps, vcfg)
                q = dict(analysis["quality"])
                dq = compute_quality([], {}, doc)
                q.update({k: v for k, v in dq.items() if k.startswith("doc_")})
            else:
                q = compute_quality(vid.frames, vid.meta, doc)
            prepared = PreparedSession(session_id=session_id, document_image=doc, frames=vid.frames,
                                       frame_times_ms=vid.times_ms, video_meta=vid.meta,
                                       challenge=steps, quality=q, analysis=analysis)
            outputs: list[ModuleOutput] = []
            for name, mod in self.modules.items():
                try:
                    outputs.append(mod.predict(prepared))
                except Exception as e:  # a failing module must not kill the session
                    log.error("module_failed", module=name, error=str(e))
                    outputs.append(ModuleOutput(module=name, status="failed", model_version=mod.version,
                                                message=f"{type(e).__name__}: {e}"))
            result = decide(outputs, self.policy, self.fusion, q)
            self._persist(session_id, outputs, q, vid.meta, doc_meta, result, analysis)
            log.info("session_processed", session_id=session_id, decision=result["decision"],
                     bundle_version=self.bundle_version)
        except MediaError as e:
            self._fail(session_id, f"Unreadable upload: {e}")
        except Exception as e:
            log.error("session_failed", session_id=session_id, error=str(e), tb=traceback.format_exc())
            self._fail(session_id, f"Internal error: {type(e).__name__}")

    def _persist(self, session_id, outputs, q, video_meta, doc_meta, result, analysis=None) -> None:
        import numpy as np

        from backend.app.db import EmbeddingRow, IdentityTemplate
        now = dt.datetime.now(dt.timezone.utc)
        with self.sm() as db:
            s = db.get(KYCSession, session_id)
            if s is None:
                return
            for o in outputs:
                db.add(ModuleOutputRow(session_id=session_id, module=o.module, status=o.status, score=o.score,
                                       raw_score=o.raw_score, confidence=o.confidence, latency_ms=o.latency_ms,
                                       model_version=o.model_version, features=_jsonable(o.features),
                                       explanations=_jsonable(o.explanations), message=o.message))
            if analysis:
                # artifact-space embeddings for novelty detection (never identity)
                from monitoring.novelty import ARTIFACT_SPACE, artifact_vector
                for space, vec in ((ARTIFACT_SPACE, artifact_vector(analysis["m2_vec"], analysis["m2"])),
                                   ("m2_clip", analysis["m2_vec"].get("clip_mean")),
                                   ("m3_lbp", analysis["m3_vec"].get("lbp")),
                                   ("m1_spectrum", (analysis.get("doc") or {}).get("m1_vec", {}).get("spectrum"))):
                    if vec is not None:
                        v = np.asarray(vec, "<f4")
                        db.add(EmbeddingRow(session_id=session_id, space=space, dim=len(v), vector=v.tobytes()))
                if len(analysis.get("arc_embs", [])):
                    t = np.asarray(analysis["arc_embs"], "<f4").mean(axis=0)
                    db.add(IdentityTemplate(session_id=session_id, expires_at=now + self.media_ttl,
                                            vector_encrypted=self.store.keyring.encrypt(
                                                t.tobytes(), f"{session_id}/identity_template".encode())))
            # NaN is not valid JSON; store as None
            db.add(QualityRow(session_id=session_id, values=_jsonable(q)))
            s.video_meta = video_meta
            s.bundle_version = self.bundle_version
            s.decision = result["decision"]
            s.decision_reason = result["reason"]
            s.fused_score = result["fused_score"]
            s.fused_uncertainty = result["fused_uncertainty"]
            s.fusion_strategy = (result.get("weights") or {}).get("strategy")
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
