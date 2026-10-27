"""Evidence-backed recommendations (ARCHITECTURE section 10).

Sensitivity S[module, feature, direction] = AUC of the module on reference sessions whose quality feature is in
the tail the current window moved into (lowest/highest tertile) minus its overall AUC. This is an OBSERVATIONAL
estimate from the reference window; the interventional estimate (controlled corruptions, E5a) replaces it once
available (`source` field says which one was used).
"""
from __future__ import annotations

import numpy as np

MODULES = ("M1", "M2", "M3", "M4", "M5")
ACTIONS = {
    "blur_laplacian_var": "augment training with Gaussian/motion blur and recalibrate",
    "tenengrad": "augment training with blur / defocus and recalibrate",
    "brightness_mean": "augment with gamma/low-light and exposure changes; check illumination guidance in the capture UI",
    "contrast_rms": "augment with contrast reduction / haze",
    "clipped_frac": "augment with over-exposure; show an exposure warning in the capture UI",
    "noise_sigma": "augment with sensor noise (Gaussian + Poisson)",
    "bits_per_pixel": "augment with stronger H.264/JPEG compression (lower bitrate)",
    "fps_measured": "augment with frame dropping / lower frame rates; re-tune rPPG window",
    "video_height": "augment with down-scaling to the new resolution",
    "face_height_frac": "augment with smaller/larger face crops; add a 'move closer' prompt",
    "head_yaw_std": "augment with more head-pose variation",
    "doc_blur_laplacian_var": "augment document images with blur / recapture conditions",
}


def sensitivity_table(Q: np.ndarray, names: list[str], P: np.ndarray, y: np.ndarray) -> dict:
    from sklearn.metrics import roc_auc_score
    out: dict = {}
    for i, m in enumerate(MODULES):
        ok = np.isfinite(P[:, i])
        if ok.sum() < 30 or len(set(y[ok])) < 2:
            continue
        base = roc_auc_score(y[ok], P[ok, i])
        for j, f in enumerate(names):
            q = Q[:, j]
            fin = ok & np.isfinite(q)
            if fin.sum() < 60:
                continue
            lo, hi = np.quantile(q[fin], [1 / 3, 2 / 3])
            for d, msk in (("low", fin & (q <= lo)), ("high", fin & (q >= hi))):
                if msk.sum() >= 20 and len(set(y[msk])) == 2:
                    out.setdefault(m, {}).setdefault(f, {})[d] = {"auc": float(roc_auc_score(y[msk], P[msk, i])),
                                                                   "delta_auc": float(roc_auc_score(y[msk], P[msk, i]) - base),
                                                                   "n": int(msk.sum())}
    return {"source": "observational (reference window tertiles)", "table": out}


def recommend(window_report: dict, sens: dict, window_label: str, bundle_version: str) -> list[dict]:
    recs = []
    lab = window_report["label"]
    if lab in ("QUALITY_DRIFT", "MIXED", "CONCEPT_DRIFT"):
        for f in window_report["top_drifting_features"][:3]:
            if not f["drifted"]:
                continue
            d = "low" if (f.get("mean_shift_sd") or 0) < 0 else "high"
            impacts = []
            for m, t in sens.get("table", {}).items():
                e = t.get(f["feature"], {}).get(d)
                if e:
                    impacts.append((e["delta_auc"], m, e))
            impacts.sort()
            msg = (f"Drift detected in {f['feature']} (PSI {f['psi']:.2f} [crit {f['psi_crit']:.3f}], "
                   f"KS q={f['ks_q']:.1e}, shift {f['mean_shift_sd']:+.2f} SD, window {window_label}, bundle {bundle_version}).")
            if impacts and impacts[0][0] < -0.01:
                dauc, m, e = impacts[0]
                msg += (f" {m} AUC on reference sessions with {d} {f['feature']} is {e['auc']:.3f} "
                        f"({dauc:+.3f} vs overall; {sens['source']}). Recommended: fine-tune / "
                        f"{ACTIONS.get(f['feature'], 'augment with the drifted condition')} for {m}.")
                target = m
            else:
                msg += " No module shows a measurable AUC drop for this condition on the reference window; " \
                       "monitor, and confirm with the controlled-corruption experiment (E5a)."
                target = None
            recs.append({"type": "drift", "feature": f["feature"], "module": target, "message": msg})
    if lab in ("NOVEL_ATTACK", "MIXED"):
        c = window_report["clusters"]
        big = max(c["clusters"], key=lambda x: x["size"]) if c["clusters"] else None
        nearest = max(c["outlier_nearest_class"].items(), key=lambda kv: kv[1])[0] if c["outlier_nearest_class"] else "?"
        ex = window_report["novelty_excess"]
        recs.append({"type": "novelty", "module": "M2/M3", "message":
                     (f"Novel-attack candidate: {ex['k_below_alpha']}/{ex['n']} sessions have conformal novelty "
                      f"p<{window_report['params']['alpha_novel']} (expected {ex['expected']:.0%}, binomial p={ex['p_value']:.1e}); "
                      f"coherent cluster of {big['size'] if big else 0} (silhouette {c['silhouette'] if c['silhouette'] is not None else 'n/a'}), "
                      f"nearest known class '{nearest}'. Recommended: route cluster members to manual review, label a sample, "
                      f"and add them as a new attack class when retraining M2/M3 (window {window_label}, bundle {bundle_version}).")})
    if lab == "OUTPUT_DRIFT":
        recs.append({"type": "output", "module": None, "message":
                     f"Decision/score distribution shifted without input drift or novelty (window {window_label}). "
                     "Likely a change in the attack prior or user population: re-check thresholds against the "
                     "target APCER/BPCER with recent human-review labels."})
    return recs
