"""Monitoring reference + pilot drift/novelty evaluation (milestone 5; pilot of E6/E7 on real features).

Reference window = fusion_train composite sessions from FF++ and MSU-MFSD only. Celeb-DF is excluded
entirely, so in the stream it is (a) an unseen capture domain for its real videos and (b) an UNSEEN
GENERATOR for its fakes (M2 was trained on FF++ only). Scenarios are drawn from fusion_test sessions:
  S0 stationary      FF++/MSU test sessions (known classes)                      expected NONE
  S1 domain drift    Celeb-DF real videos replace FF++/MSU genuine               expected QUALITY_ or CONCEPT_DRIFT
  S2 novel attack    S0 + 15% Celeb-DF fakes                                     expected NOVEL_ATTACK
  S3 mixed           S1 + 15% Celeb-DF fakes                                     expected MIXED
  S4 prior shift     S0 with replay attacks oversampled to 30%                   expected OUTPUT_DRIFT (not novelty)
"""
from __future__ import annotations

import json
import pickle

import numpy as np

from ml.fusion import MODULES, QUALITY_KEYS, FusionInput, quality_matrix
from monitoring.disambiguation import analyse_window
from monitoring.novelty import fit_novelty
from monitoring.recommend import recommend, sensitivity_table
from training.fusion_job import build_sessions, find_heads
from training.jobs import register
from training.paths import input_roots


def find_fusion(path: str | None):
    from pathlib import Path
    cands = [Path(path)] if path else []
    for r in input_roots():
        if r.is_dir():
            for d in range(0, 5):
                cands += list(r.glob("*/" * d + "runs/train_fusion/artifacts/fusion.pkl"))
    for c in cands:
        if c.exists():
            with open(c, "rb") as f:
                return pickle.load(f)
    raise FileNotFoundError("fusion.pkl not found")


def arrays(sess: list[dict], fusion: dict) -> dict:
    P = np.array([[s["P"].get(m, np.nan) for m in MODULES] for s in sess], float)
    C = np.array([[s["C"].get(m, np.nan) for m in MODULES] for s in sess], float)
    Q = quality_matrix([s["q"] for s in sess])
    fused, _, _ = fusion["model"].predict(FusionInput(P, C, Q))
    pol = fusion["policy"]
    dec = [pol.decide(float(f), {m: (None if not np.isfinite(P[i, j]) else float(P[i, j])) for j, m in enumerate(MODULES)},
                      {m: (None if not np.isfinite(C[i, j]) else float(C[i, j])) for j, m in enumerate(MODULES)})[0]
           for i, f in enumerate(fused)]
    return {"P": P, "C": C, "Q": Q, "A": np.stack([s["A"] for s in sess]).astype(np.float32), "fused": fused,
            "decisions": np.array(dec), "y": np.array([s["label"] for s in sess]),
            "attack_type": np.array([s["attack_type"] for s in sess]),
            "video_class": np.array([s["video_class"] for s in sess]), "source": np.array([s["video_source"] for s in sess]),
            "video_id": np.array([s["video_id"] for s in sess])}


def dedupe_videos(sess: list[dict]) -> list[dict]:
    """One session per video: doc-forgery sessions reuse their genuine video, and exact duplicate artifact
    vectors inflate MMD (off-diagonal k=1) - found when the stationary window was flagged as drift."""
    seen, out = set(), []
    for s in sess:
        if s["video_id"] not in seen:
            seen.add(s["video_id"])
            out.append(s)
    return out


def take(a: dict, idx: np.ndarray) -> dict:
    return {k: v[idx] for k, v in a.items()}


@register("build_reference")
def build_reference(cfg: dict, ctx) -> dict:
    p = cfg["params"]
    seed = cfg["seed"]
    rng = np.random.default_rng(seed)
    heads = find_heads(p.get("heads_dir"))
    fusion = find_fusion(p.get("fusion_path"))
    S = build_sessions(heads, seed)["sessions"]
    excl = set(p.get("exclude_sources_from_reference", ["celebdf"]))
    hold = set(p.get("holdout_classes", []))
    tr_all = dedupe_videos([s for s in S if s["part"] == "fusion_train" and s["video_source"] not in excl])
    tr = [s for s in tr_all if s["video_class"] not in hold]
    te = dedupe_videos([s for s in S if s["part"] == "fusion_test"])
    R, T = arrays(tr, fusion), arrays(te, fusion)
    ctx.log("reference", n=len(tr), stream_pool=len(te))
    from monitoring.novelty import select_config
    sel = select_config(R["A"], list(R["video_class"]), seed)
    ctx.log("novelty_selection", best=f"{sel['kind']}{sel['k']}/{sel['scorer']}",
            loco={k: round(v["mean_auroc"], 3) for k, v in sel["loco"].items()})
    p.setdefault("monitor", {})["scorer"] = sel["scorer"]
    nov = fit_novelty(R["A"], list(R["video_class"]), seed, k=sel["k"], kind=sel["kind"])
    sens = sensitivity_table(R["Q"], list(QUALITY_KEYS), R["P"], R["y"])
    qn = list(QUALITY_KEYS)

    # ---- sample-level novelty AUROC: Celeb-DF fakes (unseen generator) vs known-class test sessions
    from sklearn.metrics import roc_auc_score
    known = np.flatnonzero(T["source"] != "celebdf")
    known = np.flatnonzero((T["source"] != "celebdf") & ~np.isin(T["video_class"], list(hold)))
    sample = {}
    for nname, novel in (("holdout_" + "+".join(sorted(hold)), np.flatnonzero(np.isin(T["video_class"], list(hold)))),
                         ("celebdf_fakes", np.flatnonzero((T["source"] == "celebdf") & (T["y"] == 0))),
                         ("CONTROL_celebdf_genuine", np.flatnonzero((T["source"] == "celebdf") & (T["y"] == 1)))):
        if not len(novel):
            continue
        sc_known, sc_novel = nov.scores(T["A"][known]), nov.scores(T["A"][novel])
        for k in sc_known:
            s = np.r_[sc_known[k], sc_novel[k]]
            lab = np.r_[np.zeros(len(known)), np.ones(len(novel))]
            thr = np.quantile(sc_novel[k], 0.05)                      # FPR at 95% TPR
            sample.setdefault(nname, {})[k] = {"auroc": float(roc_auc_score(lab, s)),
                                               "fpr_at_95tpr": float((sc_known[k] >= thr).mean())}
    # ---- window scenarios.  fusion_test known-source SUBJECTS are split into a burn-in part (calibrates the
    #      monitor's empirical null, like the first weeks of production) and an evaluation part. Celeb-DF stays unseen.
    from monitoring.disambiguation import build_null
    W = int(p.get("window", 100))
    n_rep = int(p.get("repeats", 20))
    B = int(p.get("null_windows", 60))
    T["subject"] = np.array([s["subject"] for s in te])
    subj = np.unique(T["subject"][(T["source"] != "celebdf")])
    burn_subj = set(rng.choice(subj, size=int(len(subj) * float(p.get("burn_in_frac", 0.4))), replace=False))
    is_burn = np.array([s in burn_subj for s in T["subject"]])
    known = (T["source"] != "celebdf") & ~np.isin(T["video_class"], list(hold))
    strata, counts = np.unique(np.char.add(R["source"].astype(str), "|" + R["video_class"].astype(str)), return_counts=True)
    key_T = np.char.add(T["source"].astype(str), "|" + T["video_class"].astype(str))
    pools = {"cdf_genuine": np.flatnonzero((T["source"] == "celebdf") & (T["y"] == 1)),
             "cdf_fake": np.flatnonzero((T["source"] == "celebdf") & (T["y"] == 0)),
             "holdout_eval": np.flatnonzero(np.isin(T["video_class"], list(hold)) & ~is_burn),
             "prior_eval": np.flatnonzero(known & ~is_burn & (T["video_class"] == p.get("prior_shift_class", "deepfake_Face2Face")))}

    def stratified(n, burn: bool, genuine_from=None):
        idx = []
        for st, c in zip(strata, counts):
            k = int(round(n * c / counts.sum()))
            pool = np.flatnonzero((key_T == st) & known & (is_burn == burn))
            if genuine_from is not None and st.endswith("|bona_fide"):
                pool = pools[genuine_from]
            if k and len(pool):
                # WITHOUT replacement: duplicate rows inflate MMD (off-diagonal k=1) and bias the null upwards
                idx.append(rng.choice(pool, size=min(k, len(pool)), replace=False))
        return np.concatenate(idx)

    def draw(pool, n):
        return rng.choice(pools[pool], size=min(n, len(pools[pool])), replace=False)

    null = build_null(R, [take(T, stratified(W, burn=True)) for _ in range(B)], nov, qn, p.get("monitor", {}), seed)
    nn = int(float(p.get("novel_prevalence", 0.15)) * W)
    scen = {
        "S0_stationary": lambda: stratified(W, burn=False),
        "S1_domain_drift": lambda: stratified(W, burn=False, genuine_from="cdf_genuine"),
        "S2_novel_attack": lambda: np.r_[stratified(W - nn, burn=False), draw("holdout_eval", nn)],
        "S3_mixed": lambda: np.r_[stratified(W - nn, burn=False, genuine_from="cdf_genuine"), draw("holdout_eval", nn)],
        "S4_prior_shift": lambda: np.r_[stratified(W - int(0.3 * W), burn=False), draw("prior_eval", int(0.3 * W))],
        "S5_unseen_generator_new_domain": lambda: np.r_[stratified(W - nn, burn=False), draw("cdf_fake", nn)],
    }
    expected = {"S0_stationary": {"NONE"}, "S1_domain_drift": {"QUALITY_DRIFT", "CONCEPT_DRIFT"},
                "S2_novel_attack": {"NOVEL_ATTACK"}, "S3_mixed": {"MIXED"},
                "S4_prior_shift": {"OUTPUT_DRIFT", "CONCEPT_DRIFT"},          # a known class: must NOT be novelty
                "S5_unseen_generator_new_domain": {"NOVEL_ATTACK", "MIXED"}}
    results, examples = {}, {}
    for name, f in scen.items():
        labels, tests = [], []
        for r in range(n_rep):
            cur = take(T, f())
            rep = analyse_window(R, cur, nov, qn, p.get("monitor", {}), seed=seed + r, null=null)
            labels.append(rep["label"])
            tests.append(rep["tests"])
            if r == 0:
                rep["recommendations"] = recommend(rep, sens, f"{name}#0", p.get("bundle_name", "kyc-v0.4"))
                examples[name] = {k: v for k, v in rep.items() if not k.startswith("_")}
        rate = lambda key: float(np.mean([t[key] for t in tests]))

        # E7 baselines computed from the SAME window tests (only the decision rule differs)
        def b_ood_only(t):          # (b) novelty = elevated OOD rate; drift otherwise
            return "NOVEL_ATTACK" if t["H_novel"] else ("QUALITY_DRIFT" if t["H_Q"] else "CONCEPT_DRIFT" if t["H_A|Q"] else
                                                       "OUTPUT_DRIFT" if t["output_shift"] else "NONE")

        def b_no_quality(t):        # (f) proposed rule without the quality-conditional test: any shift is drift
            return ("NOVEL_ATTACK" if t["H_novel"] and t["coherent_cluster"] else
                    "CONCEPT_DRIFT" if (t["H_Q"] or t["H_A|Q"]) else "OUTPUT_DRIFT" if t["output_shift"] else "NONE")
        base = {"ood_rate_only": [b_ood_only(t) for t in tests], "no_quality_conditioning": [b_no_quality(t) for t in tests]}
        results[name] = {"labels": labels, "correct_frac": float(np.mean([l in expected[name] for l in labels])),
                         "baselines": {k: {"labels": v, "correct_frac": float(np.mean([l in expected[name] for l in v])),
                                           "false_novelty_frac": float(np.mean([l in ("NOVEL_ATTACK", "MIXED") for l in v]))}
                                       for k, v in base.items()},
                         "false_novelty_frac": float(np.mean([l in ("NOVEL_ATTACK", "MIXED") for l in labels])),
                         "expected": sorted(expected[name]),
                         "test_fire_rates": {k: rate(k) for k in ("H_Q", "H_novel", "coherent_cluster", "H_A|Q", "output_shift")}}
        ctx.log("scenario", name=name, correct=results[name]["correct_frac"], rates=results[name]["test_fire_rates"])
    T.pop("subject")
    T["is_burn_in"] = is_burn
    report = {"reference": {"n": len(tr), "classes": nov.meta["class_counts"], "excluded_sources": sorted(excl)},
              "null": {"type": "subject-level empirical (burn-in subjects)", "windows": B,
                       "burn_in_subjects": len(burn_subj), "eval_subjects": int(len(subj) - len(burn_subj))},
              "sample_level_novelty": sample, "holdout_classes": sorted(hold),
              "novelty_selection": {k: v for k, v in sel.items() if k != "loco"} | {"loco_mean_auroc": {k: v["mean_auroc"] for k, v in sel["loco"].items()}}, "scenarios": results,
              "macro_accuracy": float(np.mean([v["correct_frac"] for v in results.values()])),
              "macro_accuracy_baselines": {b: float(np.mean([v["baselines"][b]["correct_frac"] for v in results.values()]))
                                           for b in ("ood_rate_only", "no_quality_conditioning")},
              "window": W, "repeats": n_rep, "sensitivity_source": sens["source"]}
    ctx.save_artifact("reference.pkl", pickle.dumps({"quality_names": qn, "reference": R, "novelty": nov,
                                                     "sensitivity": sens, "stream_pool": T,
                                                     "excluded_sources": sorted(excl)}, protocol=pickle.HIGHEST_PROTOCOL))
    ctx.save_artifact("monitor_report.json", json.dumps(report, indent=2, default=float).encode())
    ctx.save_artifact("monitor_examples.json", json.dumps(examples, indent=2, default=float).encode())
    # served reference: ALL known classes (no hold-out), same exclusions
    Rfull = arrays(tr_all, fusion)
    nov_full = fit_novelty(Rfull["A"], list(Rfull["video_class"]), seed, k=sel["k"], kind=sel["kind"])
    null_full = build_null(Rfull, [take(T, stratified(W, burn=True)) for _ in range(B)], nov_full, qn, p.get("monitor", {}), seed)
    ctx.save_artifact("reference_served.pkl", pickle.dumps({"quality_names": qn, "reference": Rfull, "novelty": nov_full,
                                                            "null": null_full, "sensitivity": sensitivity_table(Rfull["Q"], qn, Rfull["P"], Rfull["y"]),
                                                            "window": W, "monitor": p.get("monitor", {}),
                                                            "stream_pool": {k: v for k, v in T.items() if k != "subject"},
                                                            "stream_pool_note": "simulate() must draw only from is_burn_in == False (burn-in subjects calibrate the null)"},
                                                           protocol=pickle.HIGHEST_PROTOCOL))
    return {"macro_accuracy": report["macro_accuracy"], "novelty_auroc": {n: {k: v["auroc"] for k, v in d.items()} for n, d in sample.items()},
            "finished": True}
