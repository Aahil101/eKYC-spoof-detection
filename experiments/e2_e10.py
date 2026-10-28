"""E2 calibration (reliability diagrams) and E10 subgroup audit.

    python -m experiments.e2_e10 --heads <train_heads artifacts> --fusion <fusion.pkl> [--figdir docs/figures]

E2: 10-bin reliability (equal width) of every module head on its test set and of the deployed fusion (F3) on the
    composite fusion_test sessions; ECE, max calibration error, Brier, with grouped-bootstrap CI for ECE.
E10: error rates at the deployed operating point (p >= 0.5 = genuine for heads) per capture subgroup.
    Demographic labels are NOT available in any dataset as extracted (FF++, Celeb-DF, MSU-MFSD and CelebA-Spoof
    records carry no ethnicity/age/gender fields here), so a demographic audit is not possible; the audit is over
    capture conditions that the datasets do annotate: CelebA-Spoof illumination + environment, MSU-MFSD device,
    and quality-tertiles (brightness / blur) on FF++ — the proxies most associated with skin-tone-related
    failures in the PAD literature. This limitation is stated in the paper.
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from ml.heads import build_matrix
from training.feature_jobs import load_records

SPEC = {"M2": ("m2", ("clip_mean", "spectrum")), "M3": ("m3", ("lbp",)), "M4": ("m4", ())}


def reliability(p, y, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    rows = []
    for b in range(bins):
        m = idx == b
        rows.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "n": int(m.sum()),
                     "mean_p": float(p[m].mean()) if m.any() else None, "frac_genuine": float(y[m].mean()) if m.any() else None})
    w = np.array([r["n"] for r in rows]) / len(p)
    gaps = np.array([abs(r["mean_p"] - r["frac_genuine"]) if r["n"] else 0 for r in rows])
    return {"bins": rows, "ece": float((w * gaps).sum()), "mce": float(gaps[w > 0].max()), "brier": float(np.mean((p - y) ** 2))}


def boot_ece(p, y, groups, n=500, seed=0):
    rng = np.random.default_rng(seed)
    g = np.asarray(groups)
    ug = np.unique(g)
    ix = {u: np.flatnonzero(g == u) for u in ug}
    v = []
    for _ in range(n):
        s = np.concatenate([ix[u] for u in rng.choice(ug, len(ug))])
        v.append(reliability(p[s], y[s])["ece"])
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def rates(p, y):
    gen, att = p[y == 1], p[y == 0]
    return {"n_genuine": int(len(gen)), "n_attack": int(len(att)),
            "BPCER": float((gen < 0.5).mean()) if len(gen) else None, "APCER": float((att >= 0.5).mean()) if len(att) else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=Path, required=True)
    ap.add_argument("--fusion", type=Path, required=True)
    ap.add_argument("--out", type=Path, default=Path("experiments/results/e2_e10"))
    ap.add_argument("--figdir", type=Path, default=Path("docs/figures"))
    a = ap.parse_args()
    heads = {m: pickle.load(open(a.heads / f"head_{m}.pkl", "rb")) for m in ("M1", "M2", "M3", "M4", "M5")}

    def pred(m, recs):
        key, vk = SPEC[m]
        X, *_ = build_matrix(recs, key, vk, heads[m].feature_names, heads[m].vec_dims)
        return heads[m].predict(X)[0]

    ff = [r for r in load_records("ffpp_c23") if r["split"] == "test" and r.get("face_frames", 0) > 0]
    cdf = [r for r in load_records("celebdf_v2") if r["split"] == "test" and r.get("face_frames", 0) > 0]
    msu = [r for r in load_records("msu_mfsd") if r["split"] == "test" and r.get("face_frames", 0) > 0]
    cs = [r for r in load_records("celeba_spoof") if r["split"] == "test"]
    sets = {"M2 | FF++ test": ("M2", ff), "M2 | Celeb-DF test": ("M2", cdf), "M3 | MSU-MFSD test": ("M3", msu),
            "M3 | CelebA-Spoof test": ("M3", cs), "M4 | MSU-MFSD test": ("M4", msu)}
    E2, P = {}, {}
    for name, (m, recs) in sets.items():
        p, y = pred(m, recs), np.array([r["label"] for r in recs])
        P[name] = (p, y, recs)
        E2[name] = reliability(p, y) | {"ece_ci95": boot_ece(p, y, [r["group"] for r in recs])}
    # deployed fusion on composite test sessions
    from training.fusion_job import build_sessions
    from training.monitor_job import arrays
    fusion = pickle.load(open(a.fusion, "rb"))
    S = [s for s in build_sessions(heads, 0)["sessions"] if s["part"] == "fusion_test"]
    T = arrays(S, fusion)
    E2["Fusion F3 | composite fusion_test"] = reliability(T["fused"], T["y"]) | {"ece_ci95": boot_ece(T["fused"], T["y"], [s["subject"] for s in S])}

    # ---------------- figure
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    for ax, (name, r) in zip(axes.ravel(), E2.items()):
        b = [x for x in r["bins"] if x["n"]]
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="perfect calibration")
        ax.plot([x["mean_p"] for x in b], [x["frac_genuine"] for x in b], "o-", label="model")
        tot = sum(x["n"] for x in b)
        ax.bar([(x["lo"] + x["hi"]) / 2 for x in b], [x["n"] / tot for x in b], width=0.09, alpha=0.25,
               label="fraction of samples")
        ax.set_title(f"{name}\nECE {r['ece']:.3f} [{r['ece_ci95'][0]:.3f}, {r['ece_ci95'][1]:.3f}]", fontsize=9)
        ax.set_xlabel("predicted P(genuine)")
        ax.set_ylabel("observed fraction genuine")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes.ravel()[0].legend(fontsize=7, loc="upper left")
    fig.tight_layout()
    a.figdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(a.figdir / "e2_reliability.png", dpi=120)

    # ---------------- E10 subgroups
    E10 = {}
    p, y, recs = P["M3 | CelebA-Spoof test"]
    # CelebA-Spoof annotates illumination/environment for SPOOF images only (live images carry code 0),
    # so these subgroups give per-condition APCER; BPCER is reported once for all live images.
    for attr in ("illumination", "environment"):
        vals = np.array([str(r.get(attr)) for r in recs])
        E10[f"M3 CelebA-Spoof APCER by spoof {attr} code (live images: code 0)"] = {
            v: rates(p[vals == v], y[vals == v]) for v in sorted(set(vals))}
    for mname in ("M3 | MSU-MFSD test", "M4 | MSU-MFSD test"):
        p, y, recs = P[mname]
        dev = np.array([str(r.get("device")) for r in recs])
        E10[f"{mname.split(' |')[0]} MSU-MFSD by device"] = {v: rates(p[dev == v], y[dev == v]) for v in sorted(set(dev))}
    p, y, recs = P["M2 | FF++ test"]
    for q in ("brightness_mean", "blur_laplacian_var"):
        x = np.array([r["quality"].get(q, np.nan) for r in recs], float)
        t1, t2 = np.nanpercentile(x, [33.3, 66.7])
        grp = np.where(x <= t1, "low", np.where(x <= t2, "mid", "high"))
        E10[f"M2 FF++ by {q} tertile"] = {g: rates(p[grp == g], y[grp == g]) for g in ("low", "mid", "high")}
    # fusion decisions per source domain at the deployed policy
    dec = T["decisions"]
    E10["Fusion policy by video source (genuine sessions)"] = {
        src: {"n": int(((T["source"] == src) & (T["y"] == 1)).sum()),
              **{d: float(np.mean(dec[(T["source"] == src) & (T["y"] == 1)] == d)) for d in ("ACCEPT", "HUMAN_REVIEW", "REJECT")}}
        for src in sorted(set(T["source"]))}

    a.out.mkdir(parents=True, exist_ok=True)
    (a.out / "e2_e10.json").write_text(json.dumps({"E2": E2, "E10": E10}, indent=2))
    L = ["## E2: calibration (10 equal-width bins; ECE with grouped-bootstrap 95% CI)", "",
         "| model / test set | n | ECE [95% CI] | MCE | Brier |", "|---|---|---|---|---|"]
    for k, r in E2.items():
        L.append(f"| {k} | {sum(b['n'] for b in r['bins'])} | {r['ece']:.3f} [{r['ece_ci95'][0]:.3f}, {r['ece_ci95'][1]:.3f}] | {r['mce']:.3f} | {r['brier']:.3f} |")
    L += ["", "Figure: `docs/figures/e2_reliability.png`.", "", "## E10: subgroup audit (operating point p >= 0.5 = genuine)", ""]
    for k, d in E10.items():
        L.append(f"**{k}**")
        L.append("")
        cols = list(next(iter(d.values())).keys())
        L.append("| subgroup | " + " | ".join(cols) + " |")
        L.append("|---" * (len(cols) + 1) + "|")
        for g, r in d.items():
            L.append(f"| {g} | " + " | ".join("-" if v is None else (f"{v:.3f}" if isinstance(v, float) else str(v)) for v in r.values()) + " |")
        L.append("")
    (a.out / "e2_e10.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L).encode("ascii", "replace").decode())


if __name__ == "__main__":
    main()
