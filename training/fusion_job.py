"""Composite KYC sessions + fusion/threshold training (milestone 4, EXPERIMENT_PLAN 0.3, E3, E4).

A composite session = (ID document record, selfie-video record, ID portrait embedding). Every part comes
from data the module heads did NOT train on, using the head's own split:
    fusion_train = heads' calibration/validation partitions   (fit fusion, thresholds, veto)
    fusion_test  = heads' test partitions                      (report only, never used for any choice)
Calibration rows are reused for fusion fitting (documented leakage: same rows calibrate heads and fit fusion).

Session types (attack_type):
    bona_fide                  real SIDTD doc + real video (FF++ / Celeb-DF / MSU) + same-identity ID portrait
    doc_forgery                fake SIDTD doc + real video + same-identity ID portrait
    deepfake_<method>          real doc + FF++/Celeb-DF fake video; ID portrait = the SOURCE identity (impersonated
                               victim, whose face the deepfake shows)
    replay_<instrument>        real doc + MSU-MFSD attack video; ID portrait = same client (victim) real video
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np

from ml.fusion import (MODULES, DecisionPolicy, FusionInput, choose_thresholds, choose_veto, evaluate_policy,
                       fit_fusion, quality_matrix)
from ml.heads import build_matrix, ece, group_split, pad_metrics
from training.feature_jobs import load_records, regroup_sidtd
from training.jobs import register
from training.paths import input_roots

SPEC = {"M1": ("m1", ("spectrum",)), "M2": ("m2", ("clip_mean", "spectrum")), "M3": ("m3", ("lbp",)),
        "M4": ("m4", ()), "M5": ("m5", ())}


def find_heads(heads_dir: str | None) -> dict:
    cands = [Path(heads_dir)] if heads_dir else []
    for r in input_roots():
        if r.is_dir():
            for d in range(0, 5):
                cands += [p for p in r.glob("*/" * d + "runs/train_heads/artifacts") if p.is_dir()]
    for c in cands:
        if all((c / f"head_{m}.pkl").exists() for m in MODULES):
            out = {}
            for m in MODULES:
                with open(c / f"head_{m}.pkl", "rb") as f:
                    out[m] = pickle.load(f)
            return out
    raise FileNotFoundError(f"head_M1..M5.pkl not found in {cands}")


def _score(art, recs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    key, vk = SPEC[art.module]
    if not recs:
        return np.zeros(0), np.zeros(0)
    X, *_ = build_matrix(recs, key, vk, art.feature_names, art.vec_dims)
    p, _, c = art.predict(X)
    return p, c


def build_sessions(heads: dict, seed: int = 0, doc_attack_frac: float = 0.3) -> dict:
    from ml.features.biometrics import match_features
    rng = np.random.default_rng(seed)
    ff = load_records("ffpp_c23")
    cdf = load_records("celebdf_v2")
    msu = load_records("msu_mfsd")
    sd = regroup_sidtd(load_records("sidtd"))

    # ---- fusion split per record, mirroring each head's partitions
    def fsplit_from(part):
        return {"cal": "fusion_train", "val": "fusion_train", "test": "fusion_test"}.get(part)

    for r in ff:
        r["_f"] = {"val": "fusion_train", "test": "fusion_test"}.get(r["split"])
    cdf_ids = sorted({r["identity"] for r in cdf if r["label"] == 1})
    id_part = dict(zip(cdf_ids, group_split(cdf_ids, seed)))          # same as M5 pair split (by ID owner)
    for r in cdf:
        owner = r["identity"] if r["label"] == 1 else (r.get("source_identity") or r["identity"])
        r["_f"] = fsplit_from(id_part.get(owner, "train"))
    ms_inner = group_split([r["group"] for r in msu], seed, (0.75, 0.25, 0.0, 0.0))
    for r, s in zip(msu, ms_inner):
        r["_f"] = "fusion_test" if r["split"] == "test" else ("fusion_train" if s == "cal" else None)
    sd_part = group_split([r["group"] for r in sd], seed)
    for r, s in zip(sd, sd_part):
        r["_f"] = fsplit_from(s)

    # ---- per-record module scores
    def attach(recs, mods):
        for m in mods:
            p, c = _score(heads[m], recs)
            for r, pi, ci in zip(recs, p, c):
                r[f"_p{m}"], r[f"_c{m}"] = float(pi), float(ci)
    videos = [r for r in ff + cdf + msu if r.get("_f") and r.get("face_frames", 0) > 0]
    attach(videos, ("M2", "M3", "M4"))
    docs = [r for r in sd if r.get("_f")]
    attach(docs, ("M1",))

    # ---- ID portraits: real videos indexed by identity
    real_by_id: dict[str, list[dict]] = {}
    for r in ff + cdf + msu:
        if r["label"] == 1 and len(r.get("arc_embs", [])) > 0:
            real_by_id.setdefault(r["identity"] if r["source"] != "ffpp" else r["identity"], []).append(r)

    def id_portrait(video, owner):
        cands = [v for v in real_by_id.get(owner, []) if v["id"] != video["id"]]
        if cands:
            v = cands[rng.integers(len(cands))]
            return v["arc_embs"][0], v.get("m5_ratios")
        same = real_by_id.get(owner, [])
        if same and len(video.get("arc_embs", [])) > 1:      # FF++: one video per identity -> frame 0 as "ID photo"
            return video["arc_embs"][0], video.get("m5_ratios")
        return None, None

    sessions = []
    for part in ("fusion_train", "fusion_test"):
        dreal = [d for d in docs if d["_f"] == part and d["label"] == 1]
        dfake = [d for d in docs if d["_f"] == part and d["label"] == 0]
        for v in [x for x in videos if x["_f"] == part]:
            if v["label"] == 1:
                owner, kinds = v["identity"], ["bona_fide"] + (["doc_forgery"] if rng.random() < doc_attack_frac else [])
            elif v["source"] == "msu":
                owner, kinds = v["identity"], [f"replay_{v['attack_type']}"]
            else:
                owner = v.get("source_identity") or v["identity"]
                kinds = [f"deepfake_{v['attack_type']}"]
            for kind in kinds:
                pool = dfake if kind == "doc_forgery" else dreal
                if not pool:
                    continue
                d = pool[rng.integers(len(pool))]
                id_emb, id_rat = id_portrait(v, owner)
                vid_embs = v["arc_embs"][1:] if (id_emb is not None and np.allclose(id_emb, v["arc_embs"][0])) \
                    else v["arc_embs"]
                m5 = None
                if id_emb is not None and len(vid_embs):
                    m5 = {"m5": {**match_features(id_emb, vid_embs, id_rat, v.get("m5_ratios") or {}), **v["m5_geo"]}}
                q = dict(v.get("quality") or {})
                q["doc_blur_laplacian_var"] = (d.get("quality") or {}).get("blur_laplacian_var")
                sessions.append({"part": part, "label": int(kind == "bona_fide"), "attack_type": kind,
                                 "video_id": v["id"], "doc_id": d["id"], "video_source": v["source"],
                                 "P": {"M1": d["_pM1"], "M2": v["_pM2"], "M3": v["_pM3"], "M4": v["_pM4"]},
                                 "C": {"M1": d["_cM1"], "M2": v["_cM2"], "M3": v["_cM3"], "M4": v["_cM4"]},
                                 "_m5": m5, "q": q})
    # M5 batched
    m5_idx = [i for i, s in enumerate(sessions) if s["_m5"] is not None]
    p5, c5 = _score(heads["M5"], [sessions[i]["_m5"] for i in m5_idx])
    for s in sessions:
        s["P"]["M5"], s["C"]["M5"] = float("nan"), float("nan")
    for i, p, c in zip(m5_idx, p5, c5):
        sessions[i]["P"]["M5"], sessions[i]["C"]["M5"] = float(p), float(c)
    for s in sessions:
        s.pop("_m5")
    return {"sessions": sessions}


def to_input(sess: list[dict]) -> tuple[FusionInput, np.ndarray, np.ndarray]:
    P = np.array([[s["P"].get(m, np.nan) for m in MODULES] for s in sess], float)
    C = np.array([[s["C"].get(m, np.nan) for m in MODULES] for s in sess], float)
    Q = quality_matrix([s["q"] for s in sess])
    return FusionInput(P, C, Q), np.array([s["label"] for s in sess]), np.array([s["attack_type"] for s in sess])


def _metrics(p, y):
    m = pad_metrics(p, y)
    return {k: m.get(k) for k in ("auc", "eer", "ece", "brier", "apcer", "bpcer", "acer")}


@register("train_fusion")
def train_fusion(cfg: dict, ctx) -> dict:
    p = cfg["params"]
    seed = cfg["seed"]
    heads = find_heads(p.get("heads_dir"))
    data = build_sessions(heads, seed)
    S = data["sessions"]
    tr = [s for s in S if s["part"] == "fusion_train"]
    te = [s for s in S if s["part"] == "fusion_test"]
    fi_tr, y_tr, at_tr = to_input(tr)
    fi_te, y_te, at_te = to_input(te)
    ctx.log("sessions", train=len(tr), test=len(te), train_genuine=int(y_tr.sum()), test_genuine=int(y_te.sum()))

    # ---- model selection on an internal split of fusion_train (never on fusion_test)
    sel = group_split([s["video_id"] for s in tr], seed, (0.7, 0.3, 0.0, 0.0))
    a, b = sel == "train", sel == "cal"
    strategies = p.get("strategies", ["F0", "F1", "F2", "F3", "F4", "F5", "F6"])
    report: dict = {"n_sessions": {"fusion_train": len(tr), "fusion_test": len(te)},
                    "attack_types": {t: int((at_te == t).sum()) for t in np.unique(at_te)},
                    "selection": {}, "test": {}, "single_module_test": {}, "leave_one_out_test": {}}
    for st in strategies:
        m = fit_fusion(st, fi_tr.subset(a), y_tr[a], seed)
        pb, _, _ = m.predict(fi_tr.subset(b))
        report["selection"][st] = _metrics(pb, y_tr[b])
    best = max(strategies, key=lambda st: (round(report["selection"][st]["auc"] or 0, 3),
                                           -(report["selection"][st]["ece"] or 1)))
    models = {}
    for st in strategies:
        models[st] = fit_fusion(st, fi_tr, y_tr, seed)
        pt, _, _ = models[st].predict(fi_te)
        report["test"][st] = _metrics(pt, y_te)
    for i, mod in enumerate(MODULES):
        ok = np.isfinite(fi_te.P[:, i])
        report["single_module_test"][mod] = {**_metrics(fi_te.P[ok, i], y_te[ok]), "coverage": float(ok.mean())}
        keep = [j for j in range(5) if j != i]
        def drop(fi):
            P, C = fi.P.copy(), fi.C.copy()
            P[:, i] = np.nan
            C[:, i] = np.nan
            return FusionInput(P, C, fi.Q)
        mm = fit_fusion(best, drop(fi_tr), y_tr, seed)
        pt, _, _ = mm.predict(drop(fi_te))
        report["leave_one_out_test"][f"without_{mod}"] = _metrics(pt, y_te)
    # ---- policy on fusion_train (data-driven thresholds + veto)
    model = models[best]
    p_tr, _, _ = model.predict(fi_tr)
    t_high, t_low = choose_thresholds(p_tr, y_tr, at_tr, float(p["apcer_max"]), float(p["bpcer_max"]))
    # veto eligibility: module AUC on fusion_train restricted to genuine + the attacks it targets
    from sklearn.metrics import roc_auc_score
    elig = []
    for mod, prefix in (("M2", "deepfake_"), ("M3", "replay_")):
        i = MODULES.index(mod)
        msk = ((y_tr == 1) | np.char.startswith(at_tr.astype(str), prefix)) & np.isfinite(fi_tr.P[:, i])
        auc = roc_auc_score(y_tr[msk], fi_tr.P[msk, i]) if len(set(y_tr[msk])) > 1 else 0
        report.setdefault("veto_eligibility", {})[mod] = {"auc_on_target_attacks_fusion_train": float(auc),
                                                          "eligible": bool(auc >= float(p["veto_min_auc"]))}
        if auc >= float(p["veto_min_auc"]):
            elig.append(mod)
    veto = choose_veto(fi_tr.P, fi_tr.C, y_tr, elig, float(p["veto_false_trigger_max"]),
                       float(p.get("veto_min_conf", 0.5)))
    pol = DecisionPolicy(t_high, t_low, veto, {k: p[k] for k in ("apcer_max", "bpcer_max", "veto_false_trigger_max")})
    pol.achieved_on_fit = evaluate_policy(pol, p_tr, fi_tr.P, fi_tr.C, y_tr, at_tr)
    p_te, w_te, u_te = model.predict(fi_te)
    report["policy"] = {"strategy": best, "t_high": t_high, "t_low": t_low, "veto": veto,
                        "fusion_train": pol.achieved_on_fit,
                        "fusion_test": evaluate_policy(pol, p_te, fi_te.P, fi_te.C, y_te, at_te)}
    no_veto = DecisionPolicy(t_high, t_low, {}, pol.targets)
    report["policy"]["fusion_test_without_veto"] = evaluate_policy(no_veto, p_te, fi_te.P, fi_te.C, y_te, at_te)
    report["mean_weights_test"] = {MODULES[j]: float(np.nanmean(w_te[:, j])) for j in range(5)}
    ctx.save_artifact("fusion.pkl", pickle.dumps({"strategy": best, "model": model, "policy": pol,
                                                  "modules": list(MODULES), "quality_keys": list(
                                                      __import__("ml.fusion", fromlist=["QUALITY_KEYS"]).QUALITY_KEYS)},
                                                 protocol=pickle.HIGHEST_PROTOCOL))
    ctx.save_artifact("fusion_report.json", json.dumps(report, indent=2, default=float).encode())
    ctx.save_artifact("sessions_test.json", json.dumps([{k: s[k] for k in ("label", "attack_type", "P", "C",
                                                                         "video_source")} for s in te],
                                                       default=float).encode())
    return {"best": best, "test_auc": report["test"][best]["auc"], "t_high": t_high, "t_low": t_low,
            "veto_modules": list(veto), "finished": True}
