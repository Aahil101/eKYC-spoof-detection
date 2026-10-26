"""Module heads: tabular classifiers on extracted features + post-hoc calibration + confidence.

* Learners: logistic regression (standardised, L2) or HistGradientBoosting (handles NaN natively).
* Calibration: temperature scaling on logits (Guo et al. [54]) or isotonic regression ([55]),
  fitted on a held-out calibration split only.
* Confidence (ARCHITECTURE 6.3, option b): (1 - binary entropy of calibrated p) x in-distribution
  factor from the kNN distance of the standardised feature vector to the training set,
  scaled by training-distance percentiles. It is model-agnostic and is EVALUATED (AUROC of
  confidence vs. correctness) rather than assumed meaningful.
* Metrics: AUC, EER, ECE (15 bins), Brier, APCER/BPCER at a BPCER-targeted threshold.
Artifacts are pickled HeadArtifact objects - load only from hash-verified bundles.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar

# ------------------------------------------------------------------ feature matrices
def build_matrix(records: list[dict], scalar_key: str, vec_keys: tuple[str, ...] = (),
                 names: list[str] | None = None, vec_dims: dict[str, int] | None = None):
    """records: dicts with record[scalar_key] (name->float) and record[f'{scalar_key}_vec'][k] arrays."""
    if names is None:
        names = sorted({k for r in records for k in (r.get(scalar_key) or {})})
    if vec_dims is None:
        vec_dims = {}
        for k in vec_keys:
            for r in records:
                v = (r.get(f"{scalar_key}_vec") or {}).get(k)
                if v is not None:
                    vec_dims[k] = len(v)
                    break
    cols = list(names) + [f"{k}[{i}]" for k in vec_keys if k in vec_dims for i in range(vec_dims[k])]
    X = np.full((len(records), len(cols)), np.nan, np.float32)
    for i, r in enumerate(records):
        s = r.get(scalar_key) or {}
        X[i, :len(names)] = [s.get(n, np.nan) if s.get(n) is not None else np.nan for n in names]
        off = len(names)
        for k in vec_keys:
            if k not in vec_dims:
                continue
            v = (r.get(f"{scalar_key}_vec") or {}).get(k)
            if v is not None and len(v) == vec_dims[k]:
                X[i, off:off + vec_dims[k]] = v
            off += vec_dims[k]
    X[~np.isfinite(X)] = np.nan
    return X, names, vec_dims, cols


# ------------------------------------------------------------------ calibration
class TemperatureScaler:
    def __init__(self):
        self.T = 1.0

    def fit(self, logits: np.ndarray, y: np.ndarray):
        def nll(t):
            z = logits / t
            return float(np.mean(np.logaddexp(0, z) - y * z))
        self.T = float(minimize_scalar(nll, bounds=(0.05, 20), method="bounded").x)
        return self

    def __call__(self, logits):
        return 1 / (1 + np.exp(-np.asarray(logits) / self.T))


class IsotonicCal:
    def __init__(self):
        from sklearn.isotonic import IsotonicRegression
        self.ir = IsotonicRegression(out_of_bounds="clip", y_min=1e-4, y_max=1 - 1e-4)

    def fit(self, logits, y):
        self.ir.fit(logits, y)
        return self

    def __call__(self, logits):
        return self.ir.predict(np.asarray(logits))


# ------------------------------------------------------------------ metrics
def ece(p: np.ndarray, y: np.ndarray, bins: int = 15) -> float:
    edges = np.linspace(0, 1, bins + 1)
    e = 0.0
    for a, b in zip(edges[:-1], edges[1:]):
        m = (p >= a) & (p < b) if b < 1 else (p >= a) & (p <= b)
        if m.any():
            e += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(e)


def eer(scores: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, scores)
    fnr = 1 - tpr
    i = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[i] + fnr[i]) / 2), float(thr[i])


def pad_metrics(p_genuine: np.ndarray, y_genuine: np.ndarray, bpcer_target: float = 0.05) -> dict:
    """ISO/IEC 30107-3 style: threshold chosen so BPCER (genuine rejected) ~= target; report APCER."""
    from sklearn.metrics import roc_auc_score
    g, a = p_genuine[y_genuine == 1], p_genuine[y_genuine == 0]
    out = {"n": int(len(y_genuine)), "n_genuine": int(len(g)), "n_attack": int(len(a))}
    if len(g) == 0 or len(a) == 0:
        return out
    thr = float(np.quantile(g, bpcer_target))
    out.update(auc=float(roc_auc_score(y_genuine, p_genuine)), eer=eer(p_genuine, y_genuine)[0],
               ece=ece(p_genuine, y_genuine), brier=float(np.mean((p_genuine - y_genuine) ** 2)),
               thr_at_bpcer=thr, bpcer=float((g < thr).mean()), apcer=float((a >= thr).mean()))
    out["acer"] = (out["apcer"] + out["bpcer"]) / 2
    return out


# ------------------------------------------------------------------ head
@dataclass
class HeadArtifact:
    module: str
    feature_names: list[str]
    vec_dims: dict[str, int]
    vec_keys: tuple[str, ...]
    scalar_key: str
    learner: Any
    calibrator: Any
    kind: str
    feature_version: str
    train_ref: np.ndarray            # standardised subsample of training rows (for kNN confidence)
    mu: np.ndarray
    sd: np.ndarray
    dist_p50: float
    dist_p99: float
    metrics: dict = field(default_factory=dict)
    data_info: dict = field(default_factory=dict)
    keep: np.ndarray | None = None   # columns used (all-NaN / constant training columns are dropped)

    def _k(self, X: np.ndarray) -> np.ndarray:
        return X if self.keep is None else X[:, self.keep]

    def columns(self) -> list[str]:
        """Names of the columns the learner actually sees (after dropping unusable ones)."""
        full = list(self.feature_names) + [f"{k}[{i}]" for k in self.vec_keys if k in self.vec_dims
                                           for i in range(self.vec_dims[k])]
        return full if self.keep is None else [full[i] for i in self.keep]

    def contributions(self, x: np.ndarray) -> np.ndarray:
        """logreg: coef * standardised value per used column; HGB: z-score (deviation from training mean)."""
        xk = self._k(x[None])[0]
        z = (np.where(np.isnan(xk), self.mu, xk) - self.mu) / self.sd
        if self.kind == "logreg":
            sc = self.learner[0]
            zz = (np.where(np.isnan(xk), self.mu, xk) - sc.mean_) / sc.scale_
            return self.learner[-1].coef_[0] * zz
        return z

    # ---- inference
    def logits(self, X: np.ndarray) -> np.ndarray:
        X = self._k(X)
        if self.kind == "logreg":
            Xi = np.where(np.isnan(X), self.mu, X)
            return self.learner.decision_function(Xi)
        p = self.learner.predict_proba(X)[:, 1]
        p = np.clip(p, 1e-6, 1 - 1e-6)
        return np.log(p / (1 - p))

    def _knn_dist(self, X: np.ndarray, k: int = 5) -> np.ndarray:
        X = self._k(X)
        Z = (np.where(np.isnan(X), self.mu, X) - self.mu) / self.sd
        out = []
        for z in Z:
            d = np.sqrt(((self.train_ref - z) ** 2).mean(axis=1))
            out.append(np.sort(d)[min(k, len(d)) - 1])
        return np.array(out)

    def predict(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (calibrated P(genuine), raw logit, confidence)."""
        z = self.logits(X)
        p = np.clip(self.calibrator(z), 1e-6, 1 - 1e-6)
        h = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
        d = self._knn_dist(X)
        ind = np.clip(1 - (d - self.dist_p50) / max(self.dist_p99 - self.dist_p50, 1e-6), 0, 1)
        return p, z, (1 - h) * ind

    def featurize(self, record: dict) -> np.ndarray:
        X, *_ = build_matrix([record], self.scalar_key, self.vec_keys, self.feature_names, self.vec_dims)
        return X


def fit_head(module: str, X: np.ndarray, y: np.ndarray, split: np.ndarray, names, vec_dims, vec_keys,
             scalar_key: str, kind: str = "hgb", calib: str = "isotonic", seed: int = 0,
             feature_version: str = "", bpcer_target: float = 0.05) -> HeadArtifact:
    """split: array of 'train'|'cal'|'val'|'test'. y: 1 = genuine/bona fide, 0 = attack."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    tr, ca = split == "train", split == "cal"
    Xfull = X
    with np.errstate(all="ignore"):
        finite = np.isfinite(X[tr]).any(axis=0)
        varying = np.nanstd(np.where(np.isfinite(X[tr]), X[tr], np.nan), axis=0) > 1e-12
    keep = np.flatnonzero(finite & np.nan_to_num(varying, nan=False).astype(bool))
    X = X[:, keep]
    mu = np.nanmean(X[tr], axis=0)
    mu = np.where(np.isfinite(mu), mu, 0.0)
    sd = np.nanstd(X[tr], axis=0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-9), sd, 1.0)
    if kind == "logreg":
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
        learner = make_pipeline(StandardScaler(), LogisticRegression(C=0.5, max_iter=3000, class_weight="balanced"))
        learner.fit(np.where(np.isnan(X[tr]), mu, X[tr]), y[tr])
    else:
        learner = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                                 l2_regularization=1.0, early_stopping=True,
                                                 validation_fraction=0.15, random_state=seed,
                                                 class_weight="balanced")
        learner.fit(X[tr], y[tr])
    rng = np.random.default_rng(seed)
    ref_idx = rng.choice(np.flatnonzero(tr), size=min(2000, tr.sum()), replace=False)
    Z = (np.where(np.isnan(X[ref_idx]), mu, X[ref_idx]) - mu) / sd
    art = HeadArtifact(module, list(names), dict(vec_dims), tuple(vec_keys), scalar_key, learner, None, kind,
                       feature_version, Z.astype(np.float32), mu, sd, 0.0, 1.0, keep=keep)
    art.keep = None          # X is already reduced while fitting; restored below
    X_red = X
    X = Xfull
    z_cal = art.logits(X_red[ca])
    art.calibrator = (TemperatureScaler() if calib == "temperature" else IsotonicCal()).fit(z_cal, y[ca])
    # in-distribution scale from calibration rows (not train rows: those are in train_ref themselves)
    d_cal = art._knn_dist(X_red[ca][: 2000])
    art.keep = keep
    art.dist_p50, art.dist_p99 = float(np.percentile(d_cal, 50)), float(np.percentile(d_cal, 99))
    metrics: dict = {}
    for part in ("cal", "val", "test"):
        m = split == part
        if m.sum() == 0:
            continue
        p, z, conf = art.predict(X[m])
        r = pad_metrics(p, y[m], bpcer_target)
        raw = 1 / (1 + np.exp(-z))
        r["ece_uncalibrated"] = ece(raw, y[m])
        correct = ((p >= 0.5) == (y[m] == 1)).astype(int)
        if 0 < correct.sum() < len(correct):
            from sklearn.metrics import roc_auc_score
            r["confidence_auroc_correct"] = float(roc_auc_score(correct, conf))
        r["confidence_mean"] = float(conf.mean())
        metrics[part] = r
    art.metrics = metrics
    return art


def group_split(groups: list[str], seed: int = 0, frac=(0.6, 0.1, 0.1, 0.2)) -> np.ndarray:
    """Deterministic group-disjoint split by hashing group ids."""
    import hashlib
    parts = np.array(["train", "cal", "val", "test"])
    edges = np.cumsum(frac)
    out = []
    for g in groups:
        u = int(hashlib.sha256(f"{seed}:{g}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        out.append(parts[int(np.searchsorted(edges, u, side="right").clip(0, 3))])
    return np.array(out)


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return math.log(p / (1 - p)) if np.isscalar(p) else np.log(p / (1 - p))
