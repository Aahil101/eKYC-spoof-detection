"""Drift vs novelty disambiguation (ARCHITECTURE section 9 - the paper's central contribution).

For a current window W against the reference R (same model bundle):
  H_Q      P_W(Q) = P_R(Q)                         MMD on the quality vector + per-feature KS/PSI with BH
  H_novel  rate of conformal novelty p < alpha is not elevated
  H_A|Q    P_W(A | Q) = P_R(A | Q) on INLIERS       MMD in artifact space with the reference re-weighted by the
                                                    quality-only density ratio w(q) = P_W(q)/P_R(q) [64]
  coherent cluster among outliers                   HDBSCAN cluster with size >= m_min and silhouette >= s_min

SUBJECT-LEVEL EMPIRICAL NULL. KYC data is clustered (FF++: 6 videos per subject; a window is a sample of
PEOPLE), so iid permutation / binomial p-values flag every window of new people as drift. When `null`
(statistics of windows drawn from held-out reference subjects, see build_null) is given, every window
p-value is the empirical rank of the observed statistic in that null: p = (1 + #{null >= obs}) / (B + 1).

Labels: NONE, QUALITY_DRIFT, CONCEPT_DRIFT, NOVEL_ATTACK, MIXED, OUTPUT_DRIFT.
Extends MINAS-style cluster-then-decide [84] with quality conditioning and calibrated window-level tests.
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from monitoring.drift import benjamini_hochberg, chi2_rate_shift, density_ratio_weights, mmd_test, per_feature_tests
from monitoring.novelty import NoveltyModel, binomial_excess, cluster_candidates

DEFAULTS = {"alpha": 0.05, "alpha_novel": 0.05, "scorer": "mahalanobis", "min_cluster_size": 8,
            "min_silhouette": 0.1, "min_cluster_frac_of_outliers": 0.3, "n_perm": 100, "max_n": 300,
            "max_outlier_frac": 0.5, "alpha_explain": 0.01}


def window_stats(ref: dict, cur: dict, nov: NoveltyModel, quality_names: list[str], p: dict, seed: int = 0) -> dict:
    """Raw window statistics (no decision). Larger = more evidence of change."""
    feat = per_feature_tests(ref["Q"], cur["Q"], quality_names, p["alpha"])
    mq = mmd_test(ref["Q"], cur["Q"], p["n_perm"], p["max_n"], seed)
    pv = nov.pvalues(nov.scores(cur["A"]))[p["scorer"]]
    outl = pv < p["alpha_novel"]
    Zc = nov.space.transform(cur["A"])
    Zr = nov.space.transform(ref["A"])
    inl = ~outl
    if inl.sum() >= 20:
        w = density_ratio_weights(ref["Q"], cur["Q"][inl], seed)
        ma = mmd_test(Zr, Zc[inl], p["n_perm"], p["max_n"], seed + 1, weights_ref=w)
        mu = mmd_test(Zr, Zc[inl], p["n_perm"], p["max_n"], seed + 2)
    else:
        ma = mu = {"p_value": 1.0, "mmd2": 0.0, "note": "too few inliers"}
    return {"per_feature": feat, "ks": np.array([f["ks_stat"] if f["ks_stat"] is not None else np.nan for f in feat]),
            "quality_mmd": mq, "pvals": pv, "outliers": outl, "Zc": Zc, "artifact_cond": ma, "artifact_uncond": mu,
            "novel_rate": float(outl.mean())}


def build_null(ref: dict, null_pool_windows: list[dict], nov: NoveltyModel, quality_names: list[str],
               params: dict | None = None, seed: int = 0) -> dict:
    p = {**DEFAULTS, **(params or {})}
    acc = {"quality_mmd2": [], "artifact_cond_mmd2": [], "novel_rate": [], "ks": [], "psi": [], "score_ks": [], "decision_chi2": []}
    for b, cur in enumerate(null_pool_windows):
        s = window_stats(ref, cur, nov, quality_names, p, seed + 100 + b)
        acc["quality_mmd2"].append(s["quality_mmd"]["mmd2"])
        acc["artifact_cond_mmd2"].append(s["artifact_cond"]["mmd2"])
        acc["novel_rate"].append(s["novel_rate"])
        acc["ks"].append(s["ks"])
        acc["psi"].append(np.array([f["psi"] if f["psi"] is not None else np.nan for f in s["per_feature"]]))
        o = _output_stats(ref, cur)
        acc["score_ks"].append(o["score_ks"])
        acc["decision_chi2"].append(o["decision_chi2"])
    return {k: np.array(v) for k, v in acc.items()} | {"B": len(null_pool_windows)}


def _output_stats(ref: dict, cur: dict) -> dict:
    sk = float(stats.ks_2samp(ref["fused"], cur["fused"]).statistic) if len(cur.get("fused", [])) else np.nan
    rc = dict(zip(*np.unique(ref["decisions"], return_counts=True)))
    cc = dict(zip(*np.unique(cur["decisions"], return_counts=True)))
    ch = chi2_rate_shift({k: int(v) for k, v in rc.items()}, {k: int(v) for k, v in cc.items()})
    return {"score_ks": sk, "decision_chi2": ch.get("chi2", np.nan), "decision": ch}


def _emp_p(obs: float, null: np.ndarray, tail: bool = False) -> float:
    """Empirical p of obs in the subject-level null. With B null windows the smallest attainable p is 1/(B+1);
    for per-feature tests under BH that floor is too coarse (B=60, 12 features: q >= 0.2), so beyond the
    empirical support (obs > max null) a Gaussian approximation of the null is used when tail=True."""
    null = null[np.isfinite(null)]
    if not np.isfinite(obs) or len(null) == 0:
        return 1.0
    p = float((1 + np.sum(null >= obs)) / (len(null) + 1))
    if tail and obs > null.max() and null.std() > 0:
        p = min(p, float(stats.norm.sf((obs - null.mean()) / null.std())))
    return p


def _ks_p_inflated(obs: float, null_ks: np.ndarray, n: int, m: int) -> float:
    """KS p-value from the Kolmogorov limit with a design-effect inflation c estimated from the subject-level
    null (clustered sessions inflate KS by c = median(null * sqrt(nm/(n+m))) / median(Kolmogorov) ~ 0.8276).
    Within the null's support the empirical rank is used; beyond it the inflated Kolmogorov tail."""
    null_ks = null_ks[np.isfinite(null_ks)]
    if not np.isfinite(obs) or len(null_ks) == 0:
        return 1.0
    p_emp = float((1 + np.sum(null_ks >= obs)) / (len(null_ks) + 1))
    if obs <= null_ks.max():
        return p_emp
    en = np.sqrt(n * m / (n + m))
    c = max(1.0, float(np.median(null_ks) * en / 0.8276))
    return min(p_emp, float(stats.kstwobign.sf(obs * en / c)))


def analyse_window(ref: dict, cur: dict, nov: NoveltyModel, quality_names: list[str], params: dict | None = None,
                   seed: int = 0, null: dict | None = None) -> dict:
    p = {**DEFAULTS, **(params or {})}
    s = window_stats(ref, cur, nov, quality_names, p, seed)
    out: dict = {"n_ref": int(len(ref["Q"])), "n_cur": int(len(cur["Q"])), "params": p,
                 "null": "subject-level empirical" if null else "iid permutation/binomial (no null supplied)"}
    feat = s["per_feature"]
    if null:
        # per-feature empirical p from the null KS statistics, then BH
        pe = np.array([_ks_p_inflated(k, null["ks"][:, j], f["n_ref"], f["n_cur"]) if np.isfinite(k) else np.nan
                       for j, (k, f) in enumerate(zip(s["ks"], feat))])
        q = benjamini_hochberg(pe)
        psi95 = np.nanpercentile(null["psi"], 95, axis=0) if "psi" in null else np.full(len(feat), np.nan)
        for j, (f, pj, qj) in enumerate(zip(feat, pe, q)):
            f["ks_p_empirical"] = None if not np.isfinite(pj) else float(pj)
            f["ks_q"] = None if not np.isfinite(qj) else float(qj)
            f["psi_null95"] = None if not np.isfinite(psi95[j]) else float(psi95[j])
            # significance (BH over calibrated p) AND effect size beyond what new subjects produce
            f["drifted"] = bool(f["ks_q"] is not None and f["ks_q"] < p["alpha"] and f["psi"] is not None
                                and (not np.isfinite(psi95[j]) or f["psi"] > psi95[j]))
        p_q = _emp_p(s["quality_mmd"]["mmd2"], null["quality_mmd2"])
        p_aq = _emp_p(s["artifact_cond"]["mmd2"], null["artifact_cond_mmd2"])
        p_nov = _emp_p(s["novel_rate"], null["novel_rate"])
        excess = {"n": int(len(s["pvals"])), "k_below_alpha": int(s["outliers"].sum()), "rate": s["novel_rate"],
                  "expected": float(np.mean(null["novel_rate"])), "p_value": p_nov}
    else:
        p_q = s["quality_mmd"]["p_value"]
        p_aq = s["artifact_cond"]["p_value"]
        excess = binomial_excess(s["pvals"], p["alpha_novel"])
        p_nov = excess["p_value"]
    out["per_feature"] = feat
    out["quality_mmd"] = {**s["quality_mmd"], "p_value": p_q}
    out["artifact_mmd_conditional"] = {**s["artifact_cond"], "p_value": p_aq}
    out["artifact_mmd_unconditional"] = s["artifact_uncond"]
    out["novelty_excess"] = excess
    h_q = p_q < p["alpha"] and any(f["drifted"] for f in feat)
    h_novel = p_nov is not None and p_nov < p["alpha"]
    h_aq = p_aq < p["alpha"]
    # ---- coherent cluster among outliers
    outl = s["outliers"]
    Zc = s["Zc"]
    clu = cluster_candidates(Zc[outl], p["min_cluster_size"], eps=nov.radius) if outl.sum() >= p["min_cluster_size"] else \
        {"labels": np.full(int(outl.sum()), -1), "clusters": [], "silhouette": None}
    biggest = max(clu["clusters"], key=lambda c: c["size"], default=None)
    coherent = bool(biggest and biggest["size"] >= p["min_cluster_size"]
                    and biggest["size"] >= p["min_cluster_frac_of_outliers"] * max(1, outl.sum())
                    and (clu.get("silhouette") is None or clu["silhouette"] >= p["min_silhouette"]))
    # Novelty must NOT be explained by capture quality: (i) a new attack enters as a minority of traffic,
    # (ii) its outliers must not differ in quality from the window's inliers (found in E5: global blur made
    #      most sessions outliers and the first rule version reported MIXED = a false new attack).
    out_frac = float(outl.mean())
    q_expl = {"p_value": 1.0, "note": "not tested"}
    if outl.sum() >= p["min_cluster_size"] and (~outl).sum() >= p["min_cluster_size"]:
        q_expl = mmd_test(cur["Q"][~outl], cur["Q"][outl], 300, p["max_n"], seed + 3)
    explained = q_expl["p_value"] < p["alpha_explain"]     # stricter: rejecting a new attack needs strong evidence
    majority = out_frac > p["max_outlier_frac"]
    nc, nd = nov.nearest_class(Zc[outl]) if outl.any() else ([], np.zeros(0))
    out["clusters"] = {"n_outliers": int(outl.sum()), "clusters": clu["clusters"], "silhouette": clu.get("silhouette"),
                       "coherent": coherent,
                       "outlier_nearest_class": {c: int(sum(1 for x in nc if x == c)) for c in set(nc)},
                       "outlier_nearest_distance_median": float(np.median(nd)) if len(nd) else None,
                       "reference_distance_p99": float(np.quantile(nov.cal_scores[p["scorer"]], 0.99)),
                       "outlier_fraction": out_frac, "outliers_explained_by_quality": bool(explained),
                       "outlier_vs_inlier_quality_mmd": q_expl, "outliers_are_majority": bool(majority)}
    # ---- output drift (scores / decisions)
    rc = dict(zip(*np.unique(ref["decisions"], return_counts=True))) if len(ref.get("decisions", [])) else {}
    cc = dict(zip(*np.unique(cur["decisions"], return_counts=True))) if len(cur.get("decisions", [])) else {}
    out["decision_rate_shift"] = chi2_rate_shift({k: int(v) for k, v in rc.items()},
                                                 {k: int(v) for k, v in cc.items()}) if rc and cc else {"p_value": None}
    if len(ref.get("fused", [])) and len(cur.get("fused", [])):
        ks = stats.ks_2samp(ref["fused"], cur["fused"])
        out["score_ks"] = {"stat": float(ks.statistic), "p_value": float(ks.pvalue)}
    else:
        out["score_ks"] = {"p_value": None}
    if null:
        o = _output_stats(ref, cur)
        out["score_ks"]["p_value"] = _emp_p(o["score_ks"], null["score_ks"])
        out["decision_rate_shift"]["p_value"] = _emp_p(o["decision_chi2"], null["decision_chi2"])
    h_out = any(x.get("p_value") is not None and x["p_value"] < p["alpha"] for x in (out["decision_rate_shift"], out["score_ks"]))
    # ---- decision rule (ARCHITECTURE section 9 table)
    novel = h_novel and coherent and not explained and not majority
    if h_novel and coherent and (explained or majority):
        h_aq = True if not h_q else h_aq          # outliers are a quality/global shift -> drift, not a new attack
    if novel and (h_q or h_aq):
        label = "MIXED"
    elif novel:
        label = "NOVEL_ATTACK"
    elif h_aq:
        label = "CONCEPT_DRIFT"
    elif h_q:
        label = "QUALITY_DRIFT"
    elif h_out:
        label = "OUTPUT_DRIFT"
    else:
        label = "NONE"
    out["label"] = label
    out["tests"] = {"H_Q": bool(h_q), "H_novel": bool(h_novel), "coherent_cluster": coherent, "H_A|Q": bool(h_aq),
                    "novelty_explained_by_quality": bool(explained), "outliers_majority": bool(majority),
                    "output_shift": bool(h_out), "p_quality": p_q, "p_artifact_given_quality": p_aq, "p_novel": p_nov}
    supp = {"NOVEL_ATTACK": [p_nov], "MIXED": [p_nov, min(p_q, p_aq)], "CONCEPT_DRIFT": [p_aq], "QUALITY_DRIFT": [p_q],
            "OUTPUT_DRIFT": [min(x["p_value"] for x in (out["decision_rate_shift"], out["score_ks"])
                                 if x.get("p_value") is not None)] if h_out else [1.0],
            "NONE": [1 - min(p_q, p_aq, p_nov if p_nov is not None else 1)]}[label]
    out["confidence"] = float(max(0.0, 1 - min(1.0, len(supp) * max(supp))))
    out["top_drifting_features"] = sorted([f for f in feat if f.get("ks_q") is not None],
                                          key=lambda f: (-(f["drifted"]), f["ks_q"], -(f["psi"] or 0)))[:5]
    out["_outlier_mask"] = outl
    out["_cluster_labels"] = clu["labels"]
    return out
