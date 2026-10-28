"""E6b: leave-one-method-out novelty detection with the forgery-attribution embedding (ml/attribution.py).

    python -m experiments.e6_attribution [--seeds 0 1 2] [--features artifact|artifact+lbp]

For each FF++ manipulation method h: train the embedder on FF++ TRAIN records of {bona_fide} U (methods - {h});
evaluate on FF++ TEST: known = test records of the training classes, novel = test records of h.
Controls: Celeb-DF genuine test videos (capture-domain shift, NOT a new attack) and Celeb-DF fakes (unseen
generator + domain). A useful novelty space must give high AUROC for held-out methods AND low AUROC for the
genuine control. Baseline: the previous unsupervised PCA space on the same vectors (monitoring/novelty.py).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from ml.attribution import fit_embedder, ood_scores
from monitoring.novelty import artifact_vector, fit_novelty
from training.feature_jobs import load_records

METHODS = ("Deepfakes", "Face2Face", "FaceSwap", "NeuralTextures", "FaceShifter")


def vec(r, kind):
    a = artifact_vector(r["m2_vec"], r["m2"])
    if kind == "artifact+lbp":
        a = np.r_[a, np.asarray(r["m3_vec"]["lbp"], np.float32)]
    return a


def cls(r):
    return "bona_fide" if r["label"] == 1 else r["attack_type"]


def auroc(neg, pos):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(np.r_[np.zeros(len(neg)), np.ones(len(pos))], np.r_[neg, pos]))


def fpr95(neg, pos):
    return float((neg >= np.quantile(pos, 0.05)).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--features", default="artifact", choices=["artifact", "artifact+lbp"])
    ap.add_argument("--out", type=Path, default=Path("experiments/results/e6"))
    a = ap.parse_args()
    ff = [r for r in load_records("ffpp_c23") if r.get("face_frames", 0) > 0]
    cdf = [r for r in load_records("celebdf_v2") if r.get("face_frames", 0) > 0 and r["split"] == "test"]
    X = {s: np.stack([vec(r, a.features) for r in ff if r["split"] == s]) for s in ("train", "val", "test")}
    L = {s: np.array([cls(r) for r in ff if r["split"] == s]) for s in ("train", "val", "test")}
    Xc = np.stack([vec(r, a.features) for r in cdf])
    yc = np.array([r["label"] for r in cdf])
    res = {}
    for h in METHODS:
        for seed in a.seeds:
            tr = L["train"] != h
            emb = fit_embedder(X["train"][tr], L["train"][tr], seed=seed)
            Et = emb.embed(X["train"][tr])
            known = L["test"] != h
            s_known = ood_scores(emb, X["train"][tr], L["train"][tr], X["test"][known], train_E=Et)
            s_novel = ood_scores(emb, X["train"][tr], L["train"][tr], X["test"][~known], train_E=Et)
            s_cg = ood_scores(emb, X["train"][tr], L["train"][tr], Xc[yc == 1], train_E=Et)
            s_cf = ood_scores(emb, X["train"][tr], L["train"][tr], Xc[yc == 0], train_E=Et)
            # closed-set attribution accuracy on known test classes (sanity: the embedding learnt forgery traces)
            pred = np.array(emb.classes)[emb.logits(X["test"][known]).argmax(1)]
            acc = float((pred == L["test"][known]).mean())
            r = {"closed_set_acc": acc}
            for k in s_known:
                r[k] = {"auroc_heldout": auroc(s_known[k], s_novel[k]), "fpr95_heldout": fpr95(s_known[k], s_novel[k]),
                        "auroc_CONTROL_cdf_genuine": auroc(s_known[k], s_cg[k]), "auroc_cdf_fake": auroc(s_known[k], s_cf[k])}
            # baseline: unsupervised PCA space of milestone 5 (same vectors, same splits)
            nov = fit_novelty(X["train"][tr], list(L["train"][tr]), seed=seed, k=32, use_sklearn=False)
            b_known, b_novel, b_cg = nov.scores(X["test"][known]), nov.scores(X["test"][~known]), nov.scores(Xc[yc == 1])
            r["baseline_pca32"] = {k: {"auroc_heldout": auroc(b_known[k], b_novel[k]),
                                       "auroc_CONTROL_cdf_genuine": auroc(b_known[k], b_cg[k])} for k in ("mahalanobis", "knn")}
            res.setdefault(h, []).append(r)
            print(h, seed, f"acc={acc:.3f}", {k: round(r[k]["auroc_heldout"], 3) for k in ("knn", "maha", "msp", "energy")},
                  "ctrl", {k: round(r[k]["auroc_CONTROL_cdf_genuine"], 3) for k in ("knn", "maha")},
                  "pca", {k: round(v["auroc_heldout"], 3) for k, v in r["baseline_pca32"].items()}, flush=True)
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / f"e6_attribution_{a.features}.json").write_text(json.dumps(res, indent=2))
    lines = [f"## E6b: leave-one-method-out with the attribution embedding (features: {a.features}; mean ± sd over seeds {a.seeds})", "",
             "| held-out method | closed-set acc | kNN AUROC | Maha AUROC | MSP AUROC | energy AUROC | kNN FPR@95 | control: CDF genuine (kNN) | CDF fakes (kNN) | PCA32 baseline best AUROC |",
             "|---|---|---|---|---|---|---|---|---|---|"]
    ms = lambda v: f"{np.mean(v):.3f} ± {np.std(v):.3f}"
    agg = {}
    for h, runs in res.items():
        g = lambda k, m: [r[k][m] for r in runs]
        base = [max(r["baseline_pca32"][k]["auroc_heldout"] for k in ("mahalanobis", "knn")) for r in runs]
        lines.append(f"| {h} | {ms([r['closed_set_acc'] for r in runs])} | {ms(g('knn', 'auroc_heldout'))} | {ms(g('maha', 'auroc_heldout'))} | "
                     f"{ms(g('msp', 'auroc_heldout'))} | {ms(g('energy', 'auroc_heldout'))} | {ms(g('knn', 'fpr95_heldout'))} | "
                     f"{ms(g('knn', 'auroc_CONTROL_cdf_genuine'))} | {ms(g('knn', 'auroc_cdf_fake'))} | {ms(base)} |")
        for k in ("knn", "maha", "msp", "energy"):
            agg.setdefault(k, []).extend(g(k, "auroc_heldout"))
        agg.setdefault("pca_best", []).extend(base)
    lines += ["", "Mean over all held-out methods: " + ", ".join(f"{k} {np.mean(v):.3f}" for k, v in agg.items())]
    (a.out / f"e6_attribution_{a.features}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines).encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
