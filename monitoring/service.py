"""Live monitoring service: turns logged sessions into windows, runs the drift/novelty analysis against the
bundle's reference (MONITOR/reference_served.pkl), stores DriftReport rows and raises Alerts.

The reference, null and novelty model all belong to the active bundle; reports and alerts carry its version.
`simulate` streams held-out feature records (the bundle's stream pool: FF++/MSU/Celeb-DF test sessions, features
only, no media) into the log as source="simulated", so the dashboard can be exercised before live traffic exists.
"""
from __future__ import annotations

import datetime as dt
import pickle
from pathlib import Path

import numpy as np
from sqlalchemy import select

from backend.app.db import Alert, DriftReport, EmbeddingRow, KYCSession, ModuleOutputRow, QualityRow
from ml.fusion import MODULES, quality_matrix

SEVERITY = {"NONE": None, "OUTPUT_DRIFT": "low", "QUALITY_DRIFT": "medium", "CONCEPT_DRIFT": "high",
            "NOVEL_ATTACK": "critical", "MIXED": "critical"}


def load_reference(bundle_dir: Path | None) -> dict | None:
    if bundle_dir is None:
        return None
    f = bundle_dir / "MONITOR" / "reference_served.pkl"
    if not f.exists():
        return None
    with open(f, "rb") as fh:
        return pickle.load(fh)


def _session_arrays(db, sessions: list[KYCSession]) -> dict:
    ids = [s.id for s in sessions]
    mods = {(m.session_id, m.module): m for m in db.scalars(select(ModuleOutputRow).where(ModuleOutputRow.session_id.in_(ids)))}
    qual = {q.session_id: q.values for q in db.scalars(select(QualityRow).where(QualityRow.session_id.in_(ids)))}
    embs: dict = {}
    for e in db.scalars(select(EmbeddingRow).where(EmbeddingRow.session_id.in_(ids))):
        embs.setdefault(e.session_id, {})[e.space] = np.frombuffer(e.vector, "<f4")
    A, Q, fused, dec, P = [], [], [], [], []
    for s in sessions:
        from monitoring.novelty import ARTIFACT_DIM, ARTIFACT_SPACE
        e = embs.get(s.id, {})
        a = e.get(ARTIFACT_SPACE)
        A.append(a if a is not None and len(a) == ARTIFACT_DIM else np.full(ARTIFACT_DIM, np.nan))
        Q.append(qual.get(s.id) or {})
        fused.append(s.fused_score if s.fused_score is not None else np.nan)
        dec.append(s.decision or "UNAVAILABLE")
        P.append([getattr(mods.get((s.id, m)), "score", None) for m in MODULES])
    return {"A": np.array(A, np.float32), "Q": quality_matrix(Q), "fused": np.array(fused, float),
            "decisions": np.array(dec), "P": np.array(P, float)}


def run_window(sm, ref: dict, bundle_version: str, window: int | None = None, params: dict | None = None,
               source: str | None = None) -> dict:
    from monitoring.disambiguation import analyse_window
    from monitoring.recommend import recommend
    W = int(window or ref.get("window", 200))
    with sm() as db:
        q = select(KYCSession).where(KYCSession.status == "completed", KYCSession.bundle_version == bundle_version)
        if source:
            q = q.where(KYCSession.source == source)
        sessions = list(db.scalars(q.order_by(KYCSession.created_at.desc()).limit(W)))[::-1]
        if len(sessions) < max(30, W // 4):
            return {"status": "insufficient_data", "n": len(sessions), "needed": max(30, W // 4)}
        cur = _session_arrays(db, sessions)
        ok = np.isfinite(cur["A"]).all(1) & np.isfinite(cur["fused"])
        cur = {k: v[ok] for k, v in cur.items()}
        rep = analyse_window(ref["reference"], cur, ref["novelty"], ref["quality_names"],
                             {**ref.get("monitor", {}), **(params or {})}, null=ref.get("null"))
        t0, t1 = sessions[0].created_at, sessions[-1].created_at
        label = f"{t0:%Y-%m-%d %H:%M}-{t1:%H:%M}"
        rep["recommendations"] = recommend(rep, ref["sensitivity"], label, bundle_version)
        from backend.app.pipeline import _jsonable
        clean = _jsonable({k: v for k, v in rep.items() if not k.startswith("_")})
        # evidence for plots: reference vs current histograms of the top drifting features, novelty p histogram
        hist = {}
        names = ref["quality_names"]
        for f in rep["top_drifting_features"][:3]:
            j = names.index(f["feature"])
            r_, c_ = ref["reference"]["Q"][:, j], cur["Q"][:, j]
            r_, c_ = r_[np.isfinite(r_)], c_[np.isfinite(c_)]
            if len(r_) and len(c_):
                edges = np.histogram_bin_edges(np.r_[r_, c_], 20)
                hist[f["feature"]] = {"edges": edges.tolist(), "ref": (np.histogram(r_, edges)[0] / len(r_)).tolist(),
                                      "cur": (np.histogram(c_, edges)[0] / len(c_)).tolist()}
        clean["evidence_histograms"] = _jsonable(hist)
        clean["cluster_members"] = [sessions_i.id for sessions_i, m in zip([s for s, k in zip(sessions, ok) if k],
                                                                         rep["_outlier_mask"]) if m][:50]
        dr = DriftReport(window_start=t0, window_end=t1, reference_id=f"{bundle_version}/MONITOR",
                         bundle_version=bundle_version,
                         per_feature={"features": clean["per_feature"]},
                         multivariate={k: clean[k] for k in ("quality_mmd", "artifact_mmd_conditional",
                                                             "novelty_excess", "clusters", "tests", "confidence",
                                                             "evidence_histograms", "cluster_members")},
                         streaming={"decision_rate_shift": clean["decision_rate_shift"], "score_ks": clean["score_ks"]},
                         label=clean["label"])
        db.add(dr)
        sev = SEVERITY[clean["label"]]
        if sev:
            db.add(Alert(type=clean["label"].lower(), severity=sev, confidence=clean["confidence"],
                         bundle_version=bundle_version,
                         evidence={"tests": clean["tests"], "top_features": clean["top_drifting_features"][:3],
                                   "novelty_excess": clean["novelty_excess"], "clusters": clean["clusters"],
                                   "window": label},
                         recommendation="\n".join(r["message"] for r in clean["recommendations"]) or None))
        db.commit()
        clean["report_id"] = dr.id
    return {"status": "ok", "label": clean["label"], "confidence": clean["confidence"], "report": clean}


def simulate(sm, ref: dict, bundle_version: str, scenario: str, n: int = 200, seed: int = 0,
             start: dt.datetime | None = None) -> dict:
    """Insert n simulated sessions (features only) from the stream pool for a scenario."""
    from backend.app.db import IdentityTemplate  # noqa: F401  (schema import)
    pool = ref.get("stream_pool")
    if pool is None:
        raise ValueError("bundle reference has no stream pool")
    rng = np.random.default_rng(seed)
    src, vc, y = pool["source"], pool["video_class"], pool["y"]
    # windows must not reuse the burn-in subjects that calibrated the empirical null
    evalm = ~pool.get("is_burn_in", np.zeros(len(src), bool))
    known = (src != "celebdf") & evalm
    pools = {"known": np.flatnonzero(known), "cdf_genuine": np.flatnonzero((src == "celebdf") & (y == 1)),
             "cdf_fake": np.flatnonzero((src == "celebdf") & (y == 0)),
             "known_attack": np.flatnonzero(known & (y == 0))}
    pick = lambda k, m: rng.choice(pools[k], size=min(m, len(pools[k])), replace=False)
    R = ref["reference"]
    strata, counts = np.unique(np.char.add(R["source"].astype(str), "|" + R["video_class"].astype(str)), return_counts=True)
    key = np.char.add(src.astype(str), "|" + vc.astype(str))

    def stratified(m, genuine_from=None):
        idx = []
        for st, c in zip(strata, counts):
            k = int(round(m * c / counts.sum()))
            pool_ = pools[genuine_from] if (genuine_from and st.endswith("|bona_fide")) else np.flatnonzero((key == st) & evalm)
            if k and len(pool_):
                idx.append(rng.choice(pool_, size=min(k, len(pool_)), replace=False))
        return np.concatenate(idx)
    nn = int(0.15 * n)
    sel = {"stationary": lambda: stratified(n),
           "domain_drift": lambda: stratified(n, genuine_from="cdf_genuine"),
           "unseen_generator": lambda: np.r_[stratified(n - nn), pick("cdf_fake", nn)]}[scenario]()
    rng.shuffle(sel)
    import secrets
    with sm() as db:
        last = db.scalars(select(KYCSession.created_at).order_by(KYCSession.created_at.desc()).limit(1)).first()
        now = dt.datetime.now(dt.timezone.utc)
        if last is not None and last.tzinfo is None:
            last = last.replace(tzinfo=dt.timezone.utc)
        t = start or max(now, (last + dt.timedelta(seconds=20)) if last else now)   # batches never interleave
        for i, j in enumerate(sel):
            sid = secrets.token_hex(16)
            ts = t + dt.timedelta(seconds=20 * i)
            db.add(KYCSession(id=sid, created_at=ts, updated_at=ts, status="completed", token_hash="simulated",
                              consent_version="simulated", challenge_script=[], bundle_version=bundle_version,
                              decision=str(pool["decisions"][j]), fused_score=float(pool["fused"][j]), source="simulated",
                              fusion_strategy="simulated"))
            db.flush()          # parent row first (child rows reference it; SQLite enforces FKs)
            for k, m in enumerate(MODULES):
                pv = pool["P"][j, k]
                db.add(ModuleOutputRow(session_id=sid, module=m, status="ok" if np.isfinite(pv) else "not_applicable",
                                       score=float(pv) if np.isfinite(pv) else None,
                                       confidence=float(pool["C"][j, k]) if np.isfinite(pool["C"][j, k]) else None,
                                       model_version="simulated", features={}, explanations={}))
            qv = {name: (None if not np.isfinite(pool["Q"][j, c]) else float(pool["Q"][j, c]))
                  for c, name in enumerate(ref["quality_names"])}
            db.add(QualityRow(session_id=sid, values=_unlog(qv)))
            from monitoring.novelty import ARTIFACT_SPACE
            a = pool["A"][j].astype("<f4")
            db.add(EmbeddingRow(session_id=sid, space=ARTIFACT_SPACE, dim=len(a), vector=a.tobytes()))
        db.commit()
    return {"inserted": int(len(sel)), "scenario": scenario, "ground_truth_note":
            {"stationary": "known classes, unseen subjects", "domain_drift": "Celeb-DF genuine replace FF++/MSU genuine",
             "unseen_generator": "15% Celeb-DF fakes (generator not seen by M2)"}[scenario]}


def _unlog(qv: dict) -> dict:
    """Stream-pool quality values are stored log1p-transformed (ml.fusion.quality_matrix); undo for the log."""
    from ml.fusion import LOG_KEYS
    return {k: (float(np.expm1(v)) if (v is not None and k in LOG_KEYS) else v) for k, v in qv.items()}
