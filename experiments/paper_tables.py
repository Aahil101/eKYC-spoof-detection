"""Paper tables E1 (modules), E3 (fusion ablation), E4 (operating points) with seeds and grouped-bootstrap CIs.

    python -m experiments.paper_tables --feats <dir with feats/*> --heads <train_heads artifacts> --out experiments/results/tables

Inputs are the Kaggle feature shards (CODMAV_INPUT_ROOTS) and the trained heads. Deterministic given the seeds.
Bootstrap resamples GROUPS (subjects / templates / identities), not rows, so CIs respect clustering.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

from ml.heads import build_matrix, ece, eer, group_split, pad_metrics


def boot_ci(fn, groups: np.ndarray, n: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """Point estimate + 95% percentile CI, resampling unique groups with replacement."""
    rng = np.random.default_rng(seed)
    ug = np.unique(groups)
    idx_by = {g: np.flatnonzero(groups == g) for g in ug}
    point = fn(np.arange(len(groups)))
    vals = []
    for _ in range(n):
        pick = rng.choice(ug, len(ug), replace=True)
        idx = np.concatenate([idx_by[g] for g in pick])
        try:
            v = fn(idx)
            if v is not None and np.isfinite(v):
                vals.append(v)
        except ValueError:
            pass
    lo, hi = np.percentile(vals, [2.5, 97.5]) if vals else (np.nan, np.nan)
    return float(point), float(lo), float(hi)


def _metric_fns(p, y):
    from sklearn.metrics import roc_auc_score
    def auc(i):
        return roc_auc_score(y[i], p[i]) if len(set(y[i])) == 2 else np.nan
    def eer_(i):
        return eer(p[i], y[i])[0] if len(set(y[i])) == 2 else np.nan
    def ece_(i):
        return ece(p[i], y[i])
    def apcer5(i):
        m = pad_metrics(p[i], y[i], 0.05)
        return m.get("apcer", np.nan)
    return {"AUC": auc, "EER": eer_, "ECE": ece_, "APCER@BPCER5": apcer5}


def e1_modules(heads: dict, seed: int, nboot: int) -> dict:
    from training.feature_jobs import FFPP_METHODS, _m5_pairs, load_records, regroup_sidtd
    out = {}

    def evaluate(name, art, recs, key, vk, groups):
        X, *_ = build_matrix(recs, key, vk, art.feature_names, art.vec_dims)
        p, _, _ = art.predict(X)
        y = np.array([r["label"] if "label" in r else r["_y"] for r in recs])
        out[name] = {"n": int(len(y)), "n_genuine": int(y.sum()),
                     **{k: boot_ci(f, np.asarray(groups), nboot, seed) for k, f in _metric_fns(p, y).items()}}
        return p, y

    ff = load_records("ffpp_c23")
    te = [r for r in ff if r["split"] == "test"]
    evaluate("M2 deepfake | FF++ test", heads["M2"], te, "m2", ("clip_mean", "spectrum"), [r["group"] for r in te])
    cdf = [r for r in load_records("celebdf_v2") if r["split"] == "test"]
    evaluate("M2 deepfake | Celeb-DF test (cross-dataset)", heads["M2"], cdf, "m2", ("clip_mean", "spectrum"),
             [r["identity"] for r in cdf])
    msu = load_records("msu_mfsd")
    mt = [r for r in msu if r["split"] == "test"]
    evaluate("M3 replay | MSU-MFSD test (video)", heads["M3"], mt, "m3", ("lbp",), [r["group"] for r in mt])
    cs = [r for r in load_records("celeba_spoof") if r["split"] == "test"]
    evaluate("M3 replay | CelebA-Spoof test (images)", heads["M3"], cs, "m3", ("lbp",), [r["group"] for r in cs])
    evaluate("M4 liveness | MSU-MFSD test", heads["M4"], mt, "m4", (), [r["group"] for r in mt])
    sd = regroup_sidtd(load_records("sidtd"))
    sp = group_split([r["group"] for r in sd], seed)
    st = [r for r, s in zip(sd, sp) if s == "test"]
    evaluate("M1 document | SIDTD test (seen designs)", heads["M1"], st, "m1", ("spectrum",), [r["group"] for r in st])
    real = [r for r in load_records("celebdf_v2") if r["label"] == 1]
    pairs = _m5_pairs(real, seed)
    psplit = group_split([g for _, _, g in pairs], seed)
    pt = [(q, lab, g) for (q, lab, g), s in zip(pairs, psplit) if s == "test"]
    recs = [{**q, "_y": lab} for q, lab, _ in pt]
    evaluate("M5 face match | Celeb-DF identity pairs", heads["M5"], recs, "m5", (), [g for _, _, g in pt])
    return out


def e3_fusion(heads: dict, seeds: list[int], nboot: int) -> dict:
    from ml.fusion import fit_fusion
    from training.fusion_job import build_sessions, to_input
    per_seed: dict = {}
    paired = None
    for s in seeds:
        S = build_sessions(heads, s)["sessions"]
        tr = [x for x in S if x["part"] == "fusion_train"]
        te = [x for x in S if x["part"] == "fusion_test"]
        fi_tr, y_tr, _ = to_input(tr)
        fi_te, y_te, _ = to_input(te)
        preds = {}
        for st in ("F0", "F1", "F2", "F3", "F4", "F5", "F6"):
            m = fit_fusion(st, fi_tr, y_tr, s)
            preds[st] = m.predict(fi_te)[0]
            from sklearn.metrics import roc_auc_score
            per_seed.setdefault(st, []).append({"AUC": float(roc_auc_score(y_te, preds[st])),
                                                "EER": float(eer(preds[st], y_te)[0]), "ECE": float(ece(preds[st], y_te))})
        if s == seeds[0]:
            groups = np.array([x["video_id"].split("/")[1] + "/" + str(x.get("subject", "")) for x in te])
            from sklearn.metrics import roc_auc_score
            paired = {}
            for a, b in (("F3", "F1"), ("F3", "F2"), ("F4", "F1"), ("F6", "F5")):
                paired[f"{a}-{b} AUC"] = boot_ci(lambda i: roc_auc_score(y_te[i], preds[a][i]) - roc_auc_score(y_te[i], preds[b][i])
                                                 if len(set(y_te[i])) == 2 else np.nan, np.array([x["video_id"] for x in te]),
                                                 nboot, s)
    summary = {st: {k: (float(np.mean([r[k] for r in v])), float(np.std([r[k] for r in v]))) for k in ("AUC", "EER", "ECE")}
               for st, v in per_seed.items()}
    return {"seeds": seeds, "mean_sd": summary, "paired_bootstrap_seed0": paired}


def to_markdown(e1: dict, e3: dict) -> str:
    f = lambda t: f"{t[0]:.3f} [{t[1]:.3f}, {t[2]:.3f}]"
    lines = ["## E1 — module performance (point estimate [95% grouped-bootstrap CI])", "",
             "| module / test set | n (genuine) | AUC | EER | ECE | APCER @ BPCER 5% |", "|---|---|---|---|---|---|"]
    for k, v in e1.items():
        lines.append(f"| {k} | {v['n']} ({v['n_genuine']}) | {f(v['AUC'])} | {f(v['EER'])} | {f(v['ECE'])} | {f(v['APCER@BPCER5'])} |")
    lines += ["", f"## E3 — fusion strategies, fusion_test, mean ± sd over seeds {e3['seeds']} (session composition + fitting)", "",
              "| strategy | AUC | EER | ECE |", "|---|---|---|---|"]
    for st, v in e3["mean_sd"].items():
        lines.append(f"| {st} | {v['AUC'][0]:.3f} ± {v['AUC'][1]:.3f} | {v['EER'][0]:.3f} ± {v['EER'][1]:.3f} | {v['ECE'][0]:.3f} ± {v['ECE'][1]:.3f} |")
    lines += ["", "Paired bootstrap (seed 0, resampling videos) of AUC differences:", ""]
    for k, v in (e3["paired_bootstrap_seed0"] or {}).items():
        lines.append(f"- {k}: {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]" + ("  (CI excludes 0)" if v[1] > 0 or v[2] < 0 else "  (not significant)"))
    return "\n".join(lines) + "\n"


def main():
    import pickle
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=Path("experiments/results/tables"))
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--nboot", type=int, default=1000)
    a = ap.parse_args()
    heads = {m: pickle.load(open(a.heads / f"head_{m}.pkl", "rb")) for m in ("M1", "M2", "M3", "M4", "M5")}
    a.out.mkdir(parents=True, exist_ok=True)
    e1 = e1_modules(heads, a.seeds[0], a.nboot)
    e3 = e3_fusion(heads, a.seeds, a.nboot)
    (a.out / "e1_e3.json").write_text(json.dumps({"e1": e1, "e3": e3}, indent=2, default=float))
    md = to_markdown(e1, e3)
    (a.out / "e1_e3.md").write_text(md, encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
