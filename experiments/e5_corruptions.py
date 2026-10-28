"""E5a interventional sensitivity + E5b drift detection under controlled corruptions (FF++ test subset).

    python -m experiments.e5_corruptions --heads <train_heads artifacts> --fusion <fusion.pkl> --reference <reference_served.pkl>

Needs feats/ffpp_c23 (clean) and feats/ffpp_c23__<kind><sev> shards on CODMAV_INPUT_ROOTS.
E5a: per corruption, module AUC/ECE on the SAME videos clean vs corrupted (paired), and quality-feature shifts.
E5b: windows of 200 sessions; reference = served monitor reference; stream = corrupted sessions vs clean
     sessions (same videos); detection rate, attribution hit@1/hit@3 of the expected quality feature, false alarms.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from ml.fusion import QUALITY_KEYS, FusionInput, quality_matrix
from ml.heads import build_matrix, ece
from monitoring.disambiguation import analyse_window
from monitoring.novelty import artifact_vector
from training.feature_jobs import load_records

EXPECTED_FEATURE = {"blur": {"blur_laplacian_var", "tenengrad"}, "downscale": {"blur_laplacian_var", "tenengrad"},
                    "jpeg": {"blur_laplacian_var", "tenengrad", "noise_sigma", "bits_per_pixel"},
                    "lowlight": {"brightness_mean", "contrast_rms", "clipped_frac"}, "noise": {"noise_sigma"}}
SPEC = {"M2": ("m2", ("clip_mean", "spectrum")), "M3": ("m3", ("lbp",)), "M4": ("m4", ())}


def module_scores(heads, recs):
    out = {}
    for m, (key, vk) in SPEC.items():
        X, *_ = build_matrix(recs, key, vk, heads[m].feature_names, heads[m].vec_dims)
        p, _, c = heads[m].predict(X)
        out[m] = (p, c)
    return out


def main():
    from sklearn.metrics import roc_auc_score
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=Path, required=True)
    ap.add_argument("--fusion", type=Path, required=True)
    ap.add_argument("--reference", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("experiments/results/e5"))
    ap.add_argument("--window", type=int, default=200)
    ap.add_argument("--repeats", type=int, default=20)
    a = ap.parse_args()
    heads = {m: pickle.load(open(a.heads / f"head_{m}.pkl", "rb")) for m in ("M1", "M2", "M3", "M4", "M5")}
    fusion = pickle.load(open(a.fusion, "rb"))
    ref = pickle.load(open(a.reference, "rb"))
    clean_all = {r["id"]: r for r in load_records("ffpp_c23") if r["split"] == "test"}
    sets = sorted({p.parent.name for r in __import__("training.paths", fromlist=["input_roots"]).input_roots()
                   if r.is_dir() for d in range(0, 5) for p in r.glob("*/" * d + "feats/ffpp_c23__*/shard_00000.pkl")})
    res, rng = {"E5a": {}, "E5b": {}}, np.random.default_rng(0)

    def session_arrays(recs, scores):
        Q = quality_matrix([r["quality"] for r in recs])
        P = np.full((len(recs), 5), np.nan)
        C = np.full((len(recs), 5), np.nan)
        for j, m in enumerate(("M1", "M2", "M3", "M4", "M5")):
            if m in scores:
                P[:, j], C[:, j] = scores[m]
        fused, _, _ = fusion["model"].predict(FusionInput(P, C, Q))
        dec = np.array([fusion["policy"].decide(float(f), {}, {})[0] for f in fused])
        A = np.stack([artifact_vector(r["m2_vec"], r["m2"]) for r in recs])
        return {"Q": Q, "A": A, "fused": fused, "decisions": dec}

    for tag in sets:
        kind_sev = tag.split("__")[1]
        kind = kind_sev.rstrip("0123456789")
        corr = [r for r in load_records(tag) if r["id"] in clean_all and r.get("face_frames", 0) > 0]
        clean = [clean_all[r["id"]] for r in corr]
        y = np.array([r["label"] for r in corr])
        sc_c, sc_x = module_scores(heads, clean), module_scores(heads, corr)
        e5a = {"n_videos": len(corr), "modules": {}, "quality_shift_sd": {}}
        for m in SPEC:
            pc, px = sc_c[m][0], sc_x[m][0]
            e5a["modules"][m] = {"auc_clean": float(roc_auc_score(y, pc)), "auc_corrupt": float(roc_auc_score(y, px)),
                                 "delta_auc": float(roc_auc_score(y, px) - roc_auc_score(y, pc)),
                                 "ece_clean": float(ece(pc, y)), "ece_corrupt": float(ece(px, y)),
                                 "genuine_false_reject_rate_corrupt": float((px[y == 1] < 0.5).mean()),
                                 "genuine_false_reject_rate_clean": float((pc[y == 1] < 0.5).mean())}
        Qc, Qx = quality_matrix([r["quality"] for r in clean]), quality_matrix([r["quality"] for r in corr])
        sd = np.nanstd(ref["reference"]["Q"], 0) + 1e-9
        e5a["quality_shift_sd"] = {k: float((np.nanmedian(Qx[:, j]) - np.nanmedian(Qc[:, j])) / sd[j])
                                   for j, k in enumerate(QUALITY_KEYS) if np.isfinite(Qx[:, j]).any()}
        res["E5a"][kind_sev] = e5a
        # ---- E5b: windows of corrupted sessions vs the served reference
        Sx = session_arrays(corr, {m: sc_x[m] for m in SPEC})
        Sc = session_arrays(clean, {m: sc_c[m] for m in SPEC})
        W = min(a.window, len(corr))
        hits1 = hits3 = det = fa = 0
        labels = []
        for r in range(a.repeats):
            idx = rng.choice(len(corr), W, replace=False)
            rep = analyse_window(ref["reference"], {k: v[idx] for k, v in Sx.items()}, ref["novelty"], ref["quality_names"],
                                 ref.get("monitor", {}), seed=r, null=ref.get("null"))
            labels.append(rep["label"])
            drift = rep["label"] in ("QUALITY_DRIFT", "CONCEPT_DRIFT", "MIXED")
            det += drift
            top = [f["feature"] for f in rep["top_drifting_features"] if f["drifted"]]
            hits1 += bool(top[:1]) and top[0] in EXPECTED_FEATURE[kind]
            hits3 += bool(set(top[:3]) & EXPECTED_FEATURE[kind])
            rep0 = analyse_window(ref["reference"], {k: v[idx] for k, v in Sc.items()}, ref["novelty"], ref["quality_names"],
                                  ref.get("monitor", {}), seed=r, null=ref.get("null"))
            fa += rep0["label"] != "NONE"
        res["E5b"][kind_sev] = {"window": W, "repeats": a.repeats, "detection_rate": det / a.repeats,
                                "attribution_hit@1": hits1 / a.repeats, "attribution_hit@3": hits3 / a.repeats,
                                "label_counts": {l: labels.count(l) for l in set(labels)},
                                "clean_same_videos_alarm_rate": fa / a.repeats}
        print(tag, json.dumps(res["E5b"][kind_sev]), {m: round(v["delta_auc"], 3) for m, v in e5a["modules"].items()})
    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "e5.json").write_text(json.dumps(res, indent=2))
    lines = ["## E5a: interventional sensitivity (FF++ test subset, paired clean vs corrupted, severity 2)", "",
             "| corruption | n | M2 ΔAUC | M3 ΔAUC | M4 ΔAUC | M3 genuine false-reject clean→corrupt | largest quality shift (SD) |",
             "|---|---|---|---|---|---|---|"]
    for k, v in res["E5a"].items():
        qs = max(v["quality_shift_sd"].items(), key=lambda kv: abs(kv[1]))
        mm = v["modules"]
        lines.append(f"| {k} | {v['n_videos']} | {mm['M2']['delta_auc']:+.3f} | {mm['M3']['delta_auc']:+.3f} | {mm['M4']['delta_auc']:+.3f} | "
                     f"{mm['M3']['genuine_false_reject_rate_clean']:.2f}→{mm['M3']['genuine_false_reject_rate_corrupt']:.2f} | {qs[0]} {qs[1]:+.2f} |")
    lines += ["", "## E5b: drift detection on windows of corrupted sessions (served reference + subject-level null)", "",
              "| corruption | detection rate | attribution hit@1 | hit@3 | labels | alarm rate on the same videos uncorrupted |",
              "|---|---|---|---|---|---|"]
    for k, v in res["E5b"].items():
        lines.append(f"| {k} | {v['detection_rate']:.2f} | {v['attribution_hit@1']:.2f} | {v['attribution_hit@3']:.2f} | "
                     f"{v['label_counts']} | {v['clean_same_videos_alarm_rate']:.2f} |")
    (a.out / "e5.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
