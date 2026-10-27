"""Novel-attack detection (ARCHITECTURE section 8.7-8.8; RESEARCH_NOTES [72]-[84]).

Artifact space: the M2 CLIP face embedding (512-d) concatenated with the M3 colour-LBP histogram (354-d),
standardised on the reference window and reduced by PCA (fit on reference). Never identity embeddings.

Scorers (higher = more novel):
  mahalanobis  min over known classes of the class-conditional Mahalanobis distance, shared covariance [72]
  knn          k-th nearest-neighbour distance on L2-normalised features [75]
  iforest      Isolation Forest anomaly score [77]
  lof          Local Outlier Factor (novelty mode) [78]
Every score is converted to a conformal p-value against a held-out calibration part of the reference [81].
Novel-candidate clustering: HDBSCAN on candidates in PCA space [82,83].
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class ArtifactSpace:
    """Standardise -> PCA(k) [-> shrinkage-LDA on known classes when kind="lda"]."""
    mu: np.ndarray
    sd: np.ndarray
    pca_mean: np.ndarray
    pca_comp: np.ndarray        # (k, d)
    kind: str = "pca"
    lda_mean: np.ndarray | None = None
    lda_scal: np.ndarray | None = None

    @classmethod
    def fit(cls, X: np.ndarray, k: int = 32, seed: int = 0, kind: str = "pca", labels=None) -> "ArtifactSpace":
        mu = np.nanmean(X, 0)
        mu = np.where(np.isfinite(mu), mu, 0)
        sd = np.nanstd(X, 0)
        sd = np.where(np.isfinite(sd) & (sd > 1e-9), sd, 1)
        Z = np.nan_to_num((X - mu) / sd)
        m = Z.mean(0)
        _, _, vt = np.linalg.svd(Z - m, full_matrices=False)
        sp = cls(mu, sd, m, vt[:k], kind)
        if kind == "lda":
            from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
            P = (Z - m) @ vt[:k].T
            lda = LinearDiscriminantAnalysis(solver="eigen", shrinkage="auto").fit(P, labels)
            sp.lda_mean, sp.lda_scal = lda.xbar_, lda.scalings_[:, : max(1, len(set(labels)) - 1)]
        return sp

    def transform(self, X: np.ndarray) -> np.ndarray:
        Z = np.nan_to_num((X - self.mu) / self.sd)
        P = (Z - self.pca_mean) @ self.pca_comp.T
        return P if self.kind == "pca" else (P - self.lda_mean) @ self.lda_scal


@dataclass
class NoveltyModel:
    space: ArtifactSpace
    classes: list[str]
    means: np.ndarray
    prec: np.ndarray                         # shared precision matrix
    train_Z: np.ndarray                      # reference (fit part) in PCA space, L2-normalised for kNN
    cal_scores: dict[str, np.ndarray]        # conformal calibration scores per scorer
    iforest: object = None
    lof: object = None
    knn_k: int = 10
    meta: dict = field(default_factory=dict)
    radius: float = 1.0

    # ---------------- scores
    def _maha(self, Z):
        d = np.stack([np.einsum("ij,jk,ik->i", Z - m, self.prec, Z - m) for m in self.means], 1)
        return np.sqrt(np.maximum(d, 0))

    def nearest_class(self, Z):
        d = self._maha(Z)
        i = d.argmin(1)
        return [self.classes[j] for j in i], d.min(1)

    def _knn(self, Z):
        Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
        out = []
        for i in range(0, len(Zn), 512):
            d = np.sqrt(np.maximum(2 - 2 * Zn[i:i + 512] @ self.train_Z.T, 0))
            out.append(np.sort(d, 1)[:, self.knn_k - 1])
        return np.concatenate(out) if out else np.zeros(0)

    def scores(self, X_raw: np.ndarray) -> dict[str, np.ndarray]:
        Z = self.space.transform(X_raw)
        s = {"mahalanobis": self._maha(Z).min(1), "knn": self._knn(Z)}
        if self.iforest is not None:
            s["iforest"] = -self.iforest.score_samples(Z)
        if self.lof is not None:
            s["lof"] = -self.lof.score_samples(Z)
        return s

    def pvalues(self, scores: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Conformal p-value: (1 + #{cal >= s}) / (n_cal + 1)."""
        out = {}
        for k, v in scores.items():
            cal = np.sort(self.cal_scores[k])
            ge = len(cal) - np.searchsorted(cal, v, side="left")
            out[k] = (1 + ge) / (len(cal) + 1)
        return out


def fit_novelty(X_raw: np.ndarray, labels: list[str], seed: int = 0, k: int = 32, cal_frac: float = 0.3,
                use_sklearn: bool = True, kind: str = "pca") -> NoveltyModel:
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels).astype(object)
    # classes with too few samples for a stable mean fall back to their family (replay_* -> replay)
    labels = np.array([l if (labels == l).sum() >= 15 else l.split("_")[0] for l in labels], dtype=object)
    idx = rng.permutation(len(X_raw))
    n_cal = int(len(idx) * cal_frac)
    cal, fit = idx[:n_cal], idx[n_cal:]
    space = ArtifactSpace.fit(X_raw[fit], k, seed, kind, labels[fit])
    Z = space.transform(X_raw[fit])
    classes = sorted(set(labels[fit]))
    means = np.stack([Z[labels[fit] == c].mean(0) for c in classes])
    resid = np.vstack([Z[labels[fit] == c] - means[i] for i, c in enumerate(classes)])
    cov = np.cov(resid.T) + 1e-3 * np.eye(Z.shape[1])
    prec = np.linalg.inv(cov)
    Zn = Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)
    m = NoveltyModel(space, classes, means, prec, Zn, {})
    if use_sklearn:
        from sklearn.ensemble import IsolationForest
        from sklearn.neighbors import LocalOutlierFactor
        m.iforest = IsolationForest(n_estimators=200, random_state=seed).fit(Z)
        m.lof = LocalOutlierFactor(n_neighbors=20, novelty=True).fit(Z)
    m.cal_scores = m.scores(X_raw[cal])
    m.radius = reference_radius(Z, labels[fit])
    m.meta = {"n_fit": int(len(fit)), "n_cal": int(n_cal), "pca_k": k, "space": kind,
              "class_counts": {c: int((labels[fit] == c).sum()) for c in classes}}
    return m


def binomial_excess(pvals: np.ndarray, alpha: float = 0.05) -> dict:
    """Under 'no novelty' conformal p-values are ~Uniform, so #{p < alpha} ~ Binomial(n, alpha) [81]."""
    from scipy import stats
    n = int(len(pvals))
    k = int((pvals < alpha).sum())
    return {"n": n, "k_below_alpha": k, "rate": k / n if n else None, "expected": alpha,
            "p_value": float(stats.binomtest(k, n, alpha, alternative="greater").pvalue) if n else None}


def reference_radius(Z: np.ndarray, labels, k: int = 5, q: float = 0.5) -> float:
    """Median within-known-class k-NN distance: 'as tight as a known attack family'."""
    labels = np.asarray(labels)
    ds = []
    for c in set(labels):
        P = Z[labels == c]
        if len(P) > k:
            d = np.sqrt(((P[:, None] - P[None]) ** 2).sum(-1))
            ds.append(np.sort(d, 1)[:, k])
    return float(np.quantile(np.concatenate(ds), q)) if ds else 1.0


def cluster_candidates(Z: np.ndarray, min_cluster_size: int = 8, eps: float | None = None) -> dict:
    """Density clustering of novel candidates. DBSCAN with eps = reference within-class k-NN radius (a cluster
    is 'coherent' if it is as dense as a known attack family). sklearn 1.9 HDBSCAN was evaluated first and
    recovered only 8/40 points of a tight synthetic cluster (tests/test_m5_monitoring.py), so it is not used."""
    if len(Z) < min_cluster_size:
        return {"labels": np.full(len(Z), -1), "clusters": [], "silhouette": None, "eps": eps}
    from sklearn.cluster import DBSCAN
    if eps is None:
        d = np.sqrt(((Z[:, None] - Z[None]) ** 2).sum(-1))
        eps = float(np.median(np.sort(d, 1)[:, min(5, len(Z) - 1)]))
    lab = DBSCAN(eps=eps, min_samples=min(5, min_cluster_size)).fit(Z).labels_
    clusters = []
    for c in sorted(set(lab) - {-1}):
        m = lab == c
        if m.sum() < min_cluster_size:
            lab[m] = -1
            continue
        pts = Z[m]
        cen = pts.mean(0)
        clusters.append({"cluster": int(c), "size": int(m.sum()),
                         "cohesion": float(np.mean(np.linalg.norm(pts - cen, axis=1)))})
    sil = None
    if clusters and (lab >= 0).sum() > 2 and len(set(lab)) > 1:
        from sklearn.metrics import silhouette_score
        sil = float(silhouette_score(Z, lab))
    return {"labels": lab, "clusters": clusters, "silhouette": sil, "eps": float(eps)}

CANDIDATES = [("pca", 8), ("pca", 32), ("pca", 128), ("lda", 64)]
SCORERS = ("mahalanobis", "knn", "iforest", "lof")


def select_config(X_raw: np.ndarray, labels, seed: int = 0) -> dict:
    """Leave-one-KNOWN-class-out on the reference only: for every known class c, fit on the other classes and
    measure how well each (space, scorer) separates c from held-out samples of the remaining classes.
    The best mean AUROC is chosen. No novel/test data is touched (no leakage)."""
    from sklearn.metrics import roc_auc_score
    labels = np.asarray(labels).astype(object)
    labels = np.array([l if (labels == l).sum() >= 15 else l.split("_")[0] for l in labels], dtype=object)
    rng = np.random.default_rng(seed)
    classes = [c for c in sorted(set(labels)) if c != "bona_fide" and (labels == c).sum() >= 15]
    res: dict = {}
    for kind, k in CANDIDATES:
        for c in classes:
            others = np.flatnonzero(labels != c)
            rng.shuffle(others)
            n_te = len(others) // 4
            te_known, tr_idx = others[:n_te], others[n_te:]
            try:
                m = fit_novelty(X_raw[tr_idx], list(labels[tr_idx]), seed, k, kind=kind)
            except Exception:
                continue
            sk, sn = m.scores(X_raw[te_known]), m.scores(X_raw[labels == c])
            for s in SCORERS:
                if s in sk:
                    auc = roc_auc_score(np.r_[np.zeros(len(sk[s])), np.ones(len(sn[s]))], np.r_[sk[s], sn[s]])
                    res.setdefault(f"{kind}{k}/{s}", []).append(float(auc))
    summary = {key: {"mean_auroc": float(np.mean(v)), "per_class": v} for key, v in res.items()}
    best = max(summary, key=lambda key: summary[key]["mean_auroc"])
    kind_k, scorer = best.split("/")
    kind = "lda" if kind_k.startswith("lda") else "pca"
    return {"kind": kind, "k": int(kind_k[3:]), "scorer": scorer, "loco": summary, "classes": classes}