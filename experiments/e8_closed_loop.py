"""E8 closed loop: does acting on the monitor's recommendation ("augment M2 with <corruption>") recover accuracy?

    python -m experiments.e8_closed_loop [--seeds 0 1 2]

Data: FF++ c23. The corrupted copies exist for the FF++ TEST subset (codmav-feats-ffpp-corrupt). Test SUBJECTS
are split 50/50 by a hash of the group id into
    A = "post-deployment data the operator labels and adds to training" (clean + corrupted copies)
    B = evaluation (never trained on)
E8a (module level, M2 logistic-regression head, identical recipe to training/feature_jobs.py):
    base         FF++ train (clean)                                  [the deployed head]
    +A clean     FF++ train + A clean          (controls for "just more data")
    +A rec       FF++ train + A clean + A corrupted with the RECOMMENDED corruption only
    +A all       FF++ train + A clean + A corrupted with all 5 corruptions
  evaluated on B clean and on B under each corruption. Recovery = (AUC_aug - AUC_base) / (AUC_clean_base - AUC_base).
E8b (fusion level, M2/M3/M4 + quality; M1/M5 absent): F1 / F3 / F4 fitted on FF++ val sessions (clean), evaluated
    on B clean vs corrupted: does quality gating degrade more gracefully than stacking when inputs are corrupted?
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from ml.fusion import FusionInput, fit_fusion, quality_matrix
from ml.heads import build_matrix, fit_head
from training.feature_jobs import load_records

VK = ("clip_mean", "spectrum")
SPEC = {"M2": ("m2", VK), "M3": ("m3", ("lbp",)), "M4": ("m4", ())}


def half(group: str, seed: int) -> str:
    return "A" if int(hashlib.sha256(f"e8:{seed}:{group}".encode()).hexdigest()[:8], 16) % 2 == 0 else "B"


def auc(y, p):
    from sklearn.metrics import roc_auc_score
    return float(roc_auc_score(y, p))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    ap.add_argument("--heads", type=Path, default=None, help="train_heads artifacts (M3/M4 heads for E8b)")
    ap.add_argument("--out", type=Path, default=Path("experiments/results/e8"))
    a = ap.parse_args()
    ff = [r for r in load_records("ffpp_c23") if r.get("face_frames", 0) > 0]
    from training.paths import input_roots
    tags = sorted({p.parent.name for r in input_roots() if r.is_dir() for d in range(0, 5)
                   for p in r.glob("*/" * d + "feats/ffpp_c23__*/shard_00000.pkl")})
    corr = {t.split("__")[1]: {r["id"]: r for r in load_records(t) if r.get("face_frames", 0) > 0} for t in tags}
    kinds = sorted(corr)
    test_ids = set.intersection(*[set(v) for v in corr.values()])
    test = [r for r in ff if r["split"] == "test" and r["id"] in test_ids]
    train = [r for r in ff if r["split"] == "train"]
    val = [r for r in ff if r["split"] == "val"]
    print("train", len(train), "val", len(val), "test(corrupted subset)", len(test), "corruptions", kinds, flush=True)
    heads = None
    if a.heads:
        import pickle
        heads = {m: pickle.load(open(a.heads / f"head_{m}.pkl", "rb")) for m in ("M3", "M4")}

    res = {"E8a": {}, "E8b": {}}
    for seed in a.seeds:
        hA = [r for r in test if half(r["group"], seed) == "A"]
        hB = [r for r in test if half(r["group"], seed) == "B"]
        yB = np.array([r["label"] for r in hB])

        def train_m2(extra):
            recs = train + val + extra
            X, names, vd, _ = build_matrix(recs, "m2", VK)
            y = np.array([r["label"] for r in recs])
            split = np.array(["train"] * len(train) + ["cal"] * len(val) + ["train"] * len(extra))
            return fit_head("M2", X, y, split, names, vd, VK, "m2", kind="logreg", calib="temperature", seed=seed)

        def score(art, recs):
            X, *_ = build_matrix(recs, "m2", VK, art.feature_names, art.vec_dims)
            return art.predict(X)[0]

        evalsets = {"clean": hB} | {k: [corr[k][r["id"]] for r in hB] for k in kinds}
        base = train_m2([])
        base_auc = {k: auc(yB, score(base, v)) for k, v in evalsets.items()}
        ctrl = train_m2(hA)
        ctrl_auc = {k: auc(yB, score(ctrl, v)) for k, v in evalsets.items()}
        allc = train_m2(hA + [corr[k][r["id"]] for k in kinds for r in hA])
        all_auc = {k: auc(yB, score(allc, v)) for k, v in evalsets.items()}
        rec_auc = {}
        for k in kinds:                    # the monitor's recommendation targets one corruption at a time
            art = train_m2(hA + [corr[k][r["id"]] for r in hA])
            rec_auc[k] = {"on_target": auc(yB, score(art, evalsets[k])), "on_clean": auc(yB, score(art, hB))}
        res["E8a"][seed] = {"n_A": len(hA), "n_B": len(hB), "base": base_auc, "plus_A_clean": ctrl_auc,
                            "plus_A_all": all_auc, "plus_A_recommended": rec_auc}
        print(seed, "base", {k: round(v, 3) for k, v in base_auc.items()}, flush=True)
        print(seed, "rec ", {k: round(v["on_target"], 3) for k, v in rec_auc.items()}, flush=True)

        # ---------------- E8b: fusion under corruption (M2 = deployed-recipe base head of this seed)
        if heads is not None:
            def fi_of(recs, m2art):
                P = np.full((len(recs), 5), np.nan)
                C = np.full((len(recs), 5), np.nan)
                X, *_ = build_matrix(recs, "m2", VK, m2art.feature_names, m2art.vec_dims)
                p, _, c = m2art.predict(X)
                P[:, 1], C[:, 1] = p, c
                for j, m in ((2, "M3"), (3, "M4")):
                    key, vk = SPEC[m]
                    X, *_ = build_matrix(recs, key, vk, heads[m].feature_names, heads[m].vec_dims)
                    p, _, c = heads[m].predict(X)
                    P[:, j], C[:, j] = p, c
                return FusionInput(P, C, quality_matrix([r["quality"] for r in recs]))
            fval = fi_of(val, base)
            yval = np.array([r["label"] for r in val])
            out = {}
            for strat in ("F1", "F3", "F4"):
                fm = fit_fusion(strat, fval, yval, seed)
                out[strat] = {k: auc(yB, fm.predict(fi_of(v, base))[0]) for k, v in evalsets.items()}
            res["E8b"][seed] = out
            print(seed, "fusion", {s: {k: round(x, 3) for k, x in v.items()} for s, v in out.items()}, flush=True)

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "e8.json").write_text(json.dumps(res, indent=2))
    ms = lambda v: f"{np.mean(v):.3f} ± {np.std(v):.3f}"
    S = list(res["E8a"].values())
    lines = [f"## E8a: M2 retraining as recommended (FF++ test subjects split A/B; seeds {a.seeds}; AUC on B)", "",
             "| condition | base (deployed) | + A clean (more data) | + A recommended corruption | + A all corruptions | recovery (rec) |",
             "|---|---|---|---|---|---|"]
    for k in ["clean"] + kinds:
        b = [s["base"][k] for s in S]
        c = [s["plus_A_clean"][k] for s in S]
        r = [s["plus_A_recommended"][k]["on_target"] for s in S] if k != "clean" else None
        al = [s["plus_A_all"][k] for s in S]
        if r is not None:
            rec = [(ri - bi) / max(s["base"]["clean"] - bi, 1e-6) for ri, bi, s in zip(r, b, S)]
            lines.append(f"| {k} | {ms(b)} | {ms(c)} | {ms(r)} | {ms(al)} | {ms(rec)} |")
        else:
            lines.append(f"| clean | {ms(b)} | {ms(c)} | (on clean: {ms([np.mean([v['on_clean'] for v in s['plus_A_recommended'].values()]) for s in S])}) | {ms(al)} | |")
    if res["E8b"]:
        F = list(res["E8b"].values())
        lines += ["", "## E8b: fusion of M2-M4 + quality under corruption (fitted on clean FF++ val; AUC on B)", "",
                  "| condition | F1 sum rule | F3 GBM stacking | F4 quality-gated |", "|---|---|---|---|"]
        for k in ["clean"] + kinds:
            lines.append(f"| {k} | " + " | ".join(ms([f[s][k] for f in F]) for s in ("F1", "F3", "F4")) + " |")
    (a.out / "e8.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines).encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
