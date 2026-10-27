"""Drift detection core (ARCHITECTURE section 8; RESEARCH_NOTES [59]-[71]). Implemented in-house so every
test is controllable; river/alibi-detect are only used as test oracles.

Per-feature: KS (two-sample), Wasserstein-1, PSI with a sample-size-aware critical value [66],
Jensen-Shannon distance; Benjamini-Hochberg FDR across features [65].
Multivariate: MMD^2 with RBF kernel (median heuristic) + permutation p-value [60]; classifier two-sample
test (C2ST, held-out accuracy vs Binomial(n, 1/2) null) with feature importances for attribution [61].
Streaming: ADWIN [67], Page-Hinkley [70], DDM [68], EDDM [69].
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
from scipy import stats


# ------------------------------------------------------------------ per-feature
def _clean(x):
    x = np.asarray(x, float)
    return x[np.isfinite(x)]


def psi(ref: np.ndarray, cur: np.ndarray, bins: int = 10) -> float:
    """Population stability index with reference-quantile bins (eps-smoothed)."""
    r, c = _clean(ref), _clean(cur)
    if len(r) < bins or len(c) < 2:
        return float("nan")
    edges = np.unique(np.quantile(r, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return float("nan")
    edges[0], edges[-1] = -np.inf, np.inf
    pr = np.histogram(r, edges)[0] / len(r)
    pc = np.histogram(c, edges)[0] / len(c)
    eps = 1e-4
    pr, pc = np.clip(pr, eps, None), np.clip(pc, eps, None)
    return float(np.sum((pc - pr) * np.log(pc / pr)))


def psi_critical(n_ref: int, n_cur: int, bins: int = 10, alpha: float = 0.05) -> float:
    """Approximate critical value: under H0, PSI ~ (1/n_ref + 1/n_cur) * chi2(bins-1) (Yurdakul & Naranjo [66])."""
    return float((1 / n_ref + 1 / n_cur) * stats.chi2.ppf(1 - alpha, bins - 1))


def js_distance(ref, cur, bins: int = 20) -> float:
    r, c = _clean(ref), _clean(cur)
    if len(r) < 2 or len(c) < 2:
        return float("nan")
    lo, hi = min(r.min(), c.min()), max(r.max(), c.max())
    if hi <= lo:
        return 0.0
    pr = np.histogram(r, bins, (lo, hi))[0] + 1e-9
    pc = np.histogram(c, bins, (lo, hi))[0] + 1e-9
    from scipy.spatial.distance import jensenshannon
    return float(jensenshannon(pr / pr.sum(), pc / pc.sum(), base=2))


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """BH-adjusted q-values (NaN-aware)."""
    p = np.asarray(p, float)
    q = np.full_like(p, np.nan)
    ok = np.isfinite(p)
    m = ok.sum()
    if m == 0:
        return q
    ps = p[ok]
    order = np.argsort(ps)
    ranked = ps[order] * m / (np.arange(m) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.clip(ranked, 0, 1)
    q[ok] = out
    return q


def per_feature_tests(ref: np.ndarray, cur: np.ndarray, names: list[str], alpha: float = 0.05) -> list[dict]:
    rows = []
    for j, n in enumerate(names):
        r, c = _clean(ref[:, j]), _clean(cur[:, j])
        if len(r) < 10 or len(c) < 10 or (np.ptp(r) == 0 and np.ptp(c) == 0):
            rows.append({"feature": n, "ks_stat": None, "ks_p": None, "wasserstein": None, "psi": None,
                         "psi_crit": None, "js": None, "n_ref": len(r), "n_cur": len(c)})
            continue
        ks = stats.ks_2samp(r, c)
        scale = np.std(r) + 1e-12
        rows.append({"feature": n, "ks_stat": float(ks.statistic), "ks_p": float(ks.pvalue),
                     "wasserstein": float(stats.wasserstein_distance(r, c) / scale),   # in reference-SD units
                     "psi": psi(r, c), "psi_crit": psi_critical(len(r), len(c), alpha=alpha), "js": js_distance(r, c),
                     "mean_shift_sd": float((np.mean(c) - np.mean(r)) / scale), "n_ref": len(r), "n_cur": len(c)})
    q = benjamini_hochberg(np.array([x["ks_p"] if x["ks_p"] is not None else np.nan for x in rows]))
    for x, qi in zip(rows, q):
        x["ks_q"] = None if not np.isfinite(qi) else float(qi)
        x["drifted"] = bool(x["ks_q"] is not None and x["ks_q"] < alpha and x["psi"] is not None
                            and x["psi"] > x["psi_crit"])
    return rows


# ------------------------------------------------------------------ multivariate
def _standardise(ref, cur):
    mu = np.nanmean(ref, 0)
    sd = np.nanstd(ref, 0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-12), sd, 1)
    mu = np.where(np.isfinite(mu), mu, 0)
    f = lambda X: np.nan_to_num((X - mu) / sd)
    return f(ref), f(cur)


def mmd2_rbf(X: np.ndarray, Y: np.ndarray, gamma: float) -> float:
    def k(A, B):
        d = (A ** 2).sum(1)[:, None] + (B ** 2).sum(1)[None] - 2 * A @ B.T
        return np.exp(-gamma * np.maximum(d, 0))
    m, n = len(X), len(Y)
    kxx, kyy, kxy = k(X, X), k(Y, Y), k(X, Y)
    return float((kxx.sum() - np.trace(kxx)) / (m * (m - 1)) + (kyy.sum() - np.trace(kyy)) / (n * (n - 1))
                 - 2 * kxy.mean())


def mmd_test(ref: np.ndarray, cur: np.ndarray, n_perm: int = 200, max_n: int = 400, seed: int = 0,
             weights_ref: np.ndarray | None = None) -> dict:
    """Unbiased MMD^2 with permutation p-value. weights_ref: optional importance weights to resample the
    reference (used by the quality-conditional test in disambiguation)."""
    rng = np.random.default_rng(seed)
    R, C = _standardise(ref, cur)
    if weights_ref is not None:
        w = np.clip(np.nan_to_num(weights_ref), 0, None)
        R = R[rng.choice(len(R), size=min(len(R), max_n), replace=True, p=w / w.sum())]
    elif len(R) > max_n:
        R = R[rng.choice(len(R), max_n, replace=False)]
    if len(C) > max_n:
        C = C[rng.choice(len(C), max_n, replace=False)]
    Z = np.vstack([R, C])
    d = np.sqrt(np.maximum((Z[:200, None] - Z[None, :200]) ** 2, 0).sum(-1))
    med = np.median(d[d > 0]) if (d > 0).any() else 1.0
    gamma = 1.0 / (2 * med ** 2)
    stat = mmd2_rbf(R, C, gamma)
    m = len(R)
    null = []
    for _ in range(n_perm):
        idx = rng.permutation(len(Z))
        null.append(mmd2_rbf(Z[idx[:m]], Z[idx[m:]], gamma))
    p = (1 + np.sum(np.array(null) >= stat)) / (n_perm + 1)
    return {"mmd2": stat, "p_value": float(p), "gamma": gamma, "n_ref": m, "n_cur": len(C)}


def c2st(ref: np.ndarray, cur: np.ndarray, names: list[str], seed: int = 0) -> dict:
    """Classifier two-sample test: train on half, accuracy on the other half; one-sided binomial p.
    Also returns P(current | x) on reference rows (density-ratio estimate used for reweighting)."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    rng = np.random.default_rng(seed)
    n = min(len(ref), len(cur))
    R = ref[rng.choice(len(ref), n, replace=False)]
    C = cur[rng.choice(len(cur), n, replace=False)]
    X = np.vstack([R, C])
    y = np.r_[np.zeros(n), np.ones(n)]
    idx = rng.permutation(2 * n)
    tr, te = idx[: n], idx[n:]
    clf = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15, random_state=seed)
    clf.fit(X[tr], y[tr])
    acc = float((clf.predict(X[te]) == y[te]).mean())
    k = int(round(acc * len(te)))
    p = float(stats.binomtest(k, len(te), 0.5, alternative="greater").pvalue)
    from sklearn.inspection import permutation_importance
    imp = permutation_importance(clf, X[te], y[te], n_repeats=3, random_state=seed)
    order = np.argsort(-imp.importances_mean)
    return {"accuracy": acc, "p_value": p, "n_test": int(len(te)),
            "top_features": [{"feature": names[i], "importance": float(imp.importances_mean[i])} for i in order[:8]],
            "_clf": clf}


def density_ratio_weights(ref_q: np.ndarray, cur_q: np.ndarray, seed: int = 0) -> np.ndarray:
    """w(q) = P_cur(q) / P_ref(q) for reference rows via a probabilistic classifier (Bickel et al. [64])."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    X = np.vstack([ref_q, cur_q])
    y = np.r_[np.zeros(len(ref_q)), np.ones(len(cur_q))]
    clf = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=15, random_state=seed,
                                         early_stopping=False).fit(X, y)
    p = np.clip(clf.predict_proba(ref_q)[:, 1], 0.01, 0.99)
    w = (p / (1 - p)) * (len(ref_q) / len(cur_q))
    return np.clip(w, 0, np.quantile(w, 0.99))


# ------------------------------------------------------------------ streaming detectors
@dataclass
class ADWIN:
    """ADWIN (Bifet & Gavalda 2007): adaptive window; cut when two sub-windows' means differ beyond eps_cut."""
    delta: float = 0.002
    max_window: int = 2000
    window: list = field(default_factory=list)

    def update(self, x: float) -> bool:
        self.window.append(float(x))
        if len(self.window) > self.max_window:
            self.window.pop(0)
        n = len(self.window)
        if n < 10:
            return False
        w = np.array(self.window)
        cs = np.cumsum(w)
        var = w.var()
        for i in range(5, n - 5, max(1, n // 50)):
            n0, n1 = i, n - i
            m = 1 / (1 / n0 + 1 / n1)
            dp = self.delta / n
            eps = math.sqrt(2 / m * var * math.log(2 / dp)) + 2 / (3 * m) * math.log(2 / dp)
            if abs(cs[i - 1] / n0 - (cs[-1] - cs[i - 1]) / n1) > eps:
                self.window = self.window[i:]
                return True
        return False


@dataclass
class PageHinkley:
    delta: float = 0.005
    threshold: float = 50.0
    alpha: float = 0.9999
    _mean: float = 0.0
    _n: int = 0
    _sum: float = 0.0
    _min: float = 0.0
    _sum_dn: float = 0.0
    _max_dn: float = 0.0

    def update(self, x: float) -> bool:
        self._n += 1
        self._mean += (x - self._mean) / self._n
        self._sum = self.alpha * self._sum + (x - self._mean - self.delta)
        self._min = min(self._min, self._sum)
        self._sum_dn = self.alpha * self._sum_dn + (self._mean - x - self.delta)
        self._max_dn = min(self._max_dn, self._sum_dn)
        if self._sum - self._min > self.threshold or self._sum_dn - self._max_dn > self.threshold:
            self.__init__(self.delta, self.threshold, self.alpha)
            return True
        return False


@dataclass
class DDM:
    """Error-rate control chart on a 0/1 error stream (Gama et al. 2004)."""
    min_n: int = 30
    _n: int = 0
    _p: float = 1.0
    _s: float = 0.0
    _pmin: float = float("inf")
    _smin: float = float("inf")
    warning: bool = False

    def update(self, err: int) -> bool:
        self._n += 1
        self._p += (err - self._p) / self._n
        self._s = math.sqrt(self._p * (1 - self._p) / self._n)
        if self._n < self.min_n:
            return False
        if self._p + self._s < self._pmin + self._smin:
            self._pmin, self._smin = self._p, self._s
        self.warning = self._p + self._s > self._pmin + 2 * self._smin
        if self._p + self._s > self._pmin + 3 * self._smin:
            self.__init__(self.min_n)
            return True
        return False


@dataclass
class EDDM:
    """Distance-between-errors detector (Baena-Garcia et al. 2006)."""
    alpha: float = 0.90
    min_errors: int = 30
    _t: int = 0
    _last: int = 0
    _ne: int = 0
    _m: float = 0.0
    _v: float = 0.0
    _max: float = 0.0

    def update(self, err: int) -> bool:
        self._t += 1
        if not err:
            return False
        d = self._t - self._last
        self._last = self._t
        self._ne += 1
        delta = d - self._m
        self._m += delta / self._ne
        self._v += delta * (d - self._m)
        s = math.sqrt(self._v / self._ne)
        if self._ne < self.min_errors:
            self._max = max(self._max, self._m + 2 * s)
            return False
        cur = self._m + 2 * s
        self._max = max(self._max, cur)
        if cur / self._max < self.alpha:
            self.__init__(self.alpha, self.min_errors)
            return True
        return False


def chi2_rate_shift(ref_counts: dict, cur_counts: dict) -> dict:
    keys = sorted(set(ref_counts) | set(cur_counts))
    r = np.array([ref_counts.get(k, 0) for k in keys], float)
    c = np.array([cur_counts.get(k, 0) for k in keys], float)
    if r.sum() == 0 or c.sum() == 0 or len(keys) < 2:
        return {"p_value": None}
    exp = r / r.sum() * c.sum()
    ok = exp > 0
    stat = float((((c - exp) ** 2)[ok] / exp[ok]).sum())
    return {"chi2": stat, "p_value": float(stats.chi2.sf(stat, ok.sum() - 1)),
            "ref_rates": {k: float(v / r.sum()) for k, v in zip(keys, r)},
            "cur_rates": {k: float(v / c.sum()) for k, v in zip(keys, c)}}
