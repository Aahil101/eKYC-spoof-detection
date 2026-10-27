"""Score-level fusion F0-F6, data-driven decision thresholds and veto (milestone 4, ARCHITECTURE section 6).

Inputs per session: for each module i in M1..M5 a calibrated P(genuine) p_i, a confidence c_i (both may be
missing -> NaN) and the capture-quality vector q.

  F0 mean of available p_i
  F1 fixed weighted sum of logits, weights proportional to (AUC_i - 0.5) on the fusion-train split (sum rule [49])
  F2 logistic-regression stacking on [logit p_i, c_i, missing_i]                                 ([52])
  F3 gradient-boosting stacking on [logit p_i, c_i, missing_i, q]
  F4 quality-gated mixture: w = softmax(g(q, c, missing)), z = sum_i w_i * logit p_i + b; gate g = 2-layer MLP
     ([50,53]); missing modules are masked out of the softmax
  F5 evidential Dempster-Shafer fusion of binomial opinions (b_g = c p, b_f = c (1-p), u = 1 - c),
     reduced DS rule as in TMC ([57,58]); fused P = b_g + u/2, then isotonic calibration
  F6 F5 with quality-gated discounting: c_i' = c_i * (w_i / max_j w_j) using F4's gate
Every strategy's output is calibrated (Platt, continuous) on fusion-train; reported weights are the ones actually used.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np

MODULES = ("M1", "M2", "M3", "M4", "M5")
QUALITY_KEYS = ("blur_laplacian_var", "tenengrad", "brightness_mean", "contrast_rms", "clipped_frac", "noise_sigma",
                "bits_per_pixel", "fps_measured", "video_height", "face_height_frac", "head_yaw_std",
                "doc_blur_laplacian_var")
LOG_KEYS = {"blur_laplacian_var", "tenengrad", "bits_per_pixel", "doc_blur_laplacian_var", "noise_sigma"}


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def quality_matrix(qs: list[dict]) -> np.ndarray:
    Q = np.full((len(qs), len(QUALITY_KEYS)), np.nan)
    for i, q in enumerate(qs):
        for j, k in enumerate(QUALITY_KEYS):
            v = q.get(k)
            if v is not None and np.isfinite(v):
                Q[i, j] = math.log1p(max(v, 0)) if k in LOG_KEYS else v
    return Q


@dataclass
class FusionInput:
    P: np.ndarray          # (n, 5) calibrated P(genuine), NaN = missing
    C: np.ndarray          # (n, 5) confidence, NaN = missing
    Q: np.ndarray          # (n, len(QUALITY_KEYS))

    @property
    def missing(self):
        return ~np.isfinite(self.P)

    def subset(self, m):
        return FusionInput(self.P[m], self.C[m], self.Q[m])


def ds_combine(P: np.ndarray, C: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reduced Dempster-Shafer combination of binomial opinions over available modules.
    Returns (fused P(genuine) = b_g + u/2, fused uncertainty u)."""
    n = len(P)
    bg, bf, u = np.zeros(n), np.zeros(n), np.ones(n)          # vacuous opinion
    for i in range(P.shape[1]):
        ok = np.isfinite(P[:, i]) & np.isfinite(C[:, i])
        c = np.where(ok, np.clip(C[:, i], 0, 0.999), 0.0)
        p = np.where(ok, P[:, i], 0.5)
        bg2, bf2, u2 = c * p, c * (1 - p), 1 - c
        conflict = bg * bf2 + bf * bg2
        norm = np.maximum(1 - conflict, 1e-9)
        bg, bf, u = (bg * bg2 + bg * u2 + u * bg2) / norm, (bf * bf2 + bf * u2 + u * bf2) / norm, (u * u2) / norm
    return bg + u / 2, u


class _Gate:
    """Tiny MLP gate g(q, c, missing) -> softmax weights over modules (masked by availability)."""

    def __init__(self, n_in: int, n_mod: int, hidden: int = 16, seed: int = 0):
        import torch
        torch.manual_seed(seed)
        self.torch = torch
        self.net = torch.nn.Sequential(torch.nn.Linear(n_in, hidden), torch.nn.Tanh(), torch.nn.Linear(hidden, n_mod))
        self.bias = torch.nn.Parameter(torch.zeros(1))
        self.mu = self.sd = None

    def _x(self, fi: FusionInput):
        X = np.concatenate([np.nan_to_num((fi.Q - self.mu) / self.sd), np.nan_to_num(fi.C), fi.missing.astype(float)], 1)
        return self.torch.tensor(X, dtype=self.torch.float32)

    def weights(self, fi: FusionInput) -> np.ndarray:
        with self.torch.no_grad():
            return self._w(fi).numpy()

    def _w(self, fi):
        t = self.torch
        s = self.net(self._x(fi))
        s = s.masked_fill(t.tensor(fi.missing), -1e9)
        return t.softmax(s, dim=1)

    def fit(self, fi: FusionInput, y: np.ndarray, epochs: int = 400, lr: float = 0.01, wd: float = 1e-3):
        t = self.torch
        self.mu = np.nanmean(fi.Q, 0)
        self.mu = np.where(np.isfinite(self.mu), self.mu, 0)
        self.sd = np.nanstd(fi.Q, 0)
        self.sd = np.where(np.isfinite(self.sd) & (self.sd > 1e-9), self.sd, 1)
        Z = t.tensor(np.nan_to_num(_logit(fi.P)), dtype=t.float32)
        Y = t.tensor(y, dtype=t.float32)
        pos = float(y.mean())
        wts = t.tensor(np.where(y == 1, 0.5 / pos, 0.5 / (1 - pos)), dtype=t.float32)
        opt = t.optim.Adam(list(self.net.parameters()) + [self.bias], lr=lr, weight_decay=wd)
        for _ in range(epochs):
            opt.zero_grad()
            z = (self._w(fi) * Z).sum(1) * 2.0 + self.bias       # x2: logits of weighted average are shrunk
            loss = (t.nn.functional.binary_cross_entropy_with_logits(z, Y, reduction="none") * wts).mean()
            loss.backward()
            opt.step()
        return self

    def score(self, fi):
        with self.torch.no_grad():
            Z = self.torch.tensor(np.nan_to_num(_logit(fi.P)), dtype=self.torch.float32)
            return ((self._w(fi) * Z).sum(1) * 2.0 + self.bias).numpy()


@dataclass
class FusionModel:
    strategy: str
    auc_weights: np.ndarray | None = None
    stacker: Any = None
    gate: Any = None
    calibrator: Any = None
    feature_cols: list[str] = field(default_factory=list)
    keep: np.ndarray | None = None          # F3: usable stacking columns (all-NaN/constant dropped)

    # ---------------- raw score (before isotonic)
    def _raw(self, fi: FusionInput) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns (raw score, weights actually used (n,5), uncertainty or NaN)."""
        n = len(fi.P)
        avail = (~fi.missing).astype(float)
        nanu = np.full(n, np.nan)
        if self.strategy == "F0":
            w = avail / np.maximum(avail.sum(1, keepdims=True), 1)
            return np.nansum(np.nan_to_num(fi.P) * w, 1), w, nanu
        if self.strategy == "F1":
            w = avail * self.auc_weights
            w = w / np.maximum(w.sum(1, keepdims=True), 1e-9)
            return (w * np.nan_to_num(_logit(fi.P))).sum(1), w, nanu
        if self.strategy in ("F2", "F3"):
            X = self._stack_x(fi)
            s = self.stacker.decision_function(X) if self.strategy == "F2" else _logit(self.stacker.predict_proba(X)[:, 1])
            return s, self._stack_importance(n), nanu
        if self.strategy == "F4":
            return self.gate.score(fi), self.gate.weights(fi), nanu
        if self.strategy == "F5":
            p, u = ds_combine(fi.P, fi.C)
            return p, avail * np.nan_to_num(fi.C), u
        if self.strategy == "F6":
            w = self.gate.weights(fi)
            disc = w / np.maximum(w.max(1, keepdims=True), 1e-9)
            p, u = ds_combine(fi.P, np.nan_to_num(fi.C) * disc)
            return p, avail * np.nan_to_num(fi.C) * disc, u
        raise ValueError(self.strategy)

    def _stack_x(self, fi):
        Z = np.nan_to_num(_logit(fi.P))
        C = np.nan_to_num(fi.C)
        M = fi.missing.astype(float)
        X = np.concatenate([Z, C, M] + ([fi.Q] if self.strategy == "F3" else []), 1)
        if self.strategy != "F3":
            return np.nan_to_num(X)
        return X if self.keep is None else X[:, self.keep]

    def _stack_importance(self, n):
        if self.strategy == "F2":
            co = np.abs(self.stacker[-1].coef_[0][:5])
            return np.tile(co / co.sum(), (n, 1))
        return np.full((n, 5), np.nan)       # HGB: no per-session module weights (reported as such)

    def predict(self, fi: FusionInput) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        s, w, u = self._raw(fi)
        return np.clip(self.calibrator.predict(s), 1e-6, 1 - 1e-6), w, u


class PlattCal:
    """Monotone, CONTINUOUS calibration (logistic on the raw score). Isotonic regression was tried first:
    its step function put ~16% of genuine sessions on one plateau, so no threshold between plateaus existed
    and every APCER target produced the same operating point."""

    def __init__(self, on_probability: bool):
        self.on_probability = on_probability
        self.a, self.b = 1.0, 0.0

    def _x(self, s):
        s = np.asarray(s, float)
        return _logit(s) if self.on_probability else s

    def fit(self, s, y):
        from sklearn.linear_model import LogisticRegression
        lr = LogisticRegression(C=1e4, max_iter=1000).fit(self._x(s)[:, None], y)
        self.a, self.b = float(lr.coef_[0, 0]), float(lr.intercept_[0])
        return self

    def predict(self, s):
        return 1 / (1 + np.exp(-(self.a * self._x(s) + self.b)))


def fit_fusion(strategy: str, fi: FusionInput, y: np.ndarray, seed: int = 0) -> FusionModel:
    from sklearn.metrics import roc_auc_score
    m = FusionModel(strategy)
    if strategy == "F1":
        aucs = []
        for i in range(5):
            ok = ~fi.missing[:, i]
            aucs.append(max(roc_auc_score(y[ok], fi.P[ok, i]) - 0.5, 0.0) if ok.sum() > 10 and len(set(y[ok])) > 1 else 0.0)
        m.auc_weights = np.array(aucs) + 1e-6
    elif strategy in ("F2", "F3"):
        if strategy == "F2":
            from sklearn.linear_model import LogisticRegression
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            m.stacker = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000, class_weight="balanced"))
        else:
            from sklearn.ensemble import HistGradientBoostingClassifier
            m.stacker = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, max_leaf_nodes=15,
                                                       l2_regularization=1.0, random_state=seed, class_weight="balanced")
        X = m._stack_x(fi)
        if strategy == "F3":
            with np.errstate(all="ignore"):
                ok = np.isfinite(X).sum(0) > 1
                ok &= np.nan_to_num(np.nanstd(np.where(np.isfinite(X), X, np.nan), 0)) > 1e-12
            m.keep = np.flatnonzero(ok)
            X = X[:, m.keep]
        m.stacker.fit(X, y)
    elif strategy in ("F4", "F6"):
        m.gate = _Gate(fi.Q.shape[1] + 10, 5, seed=seed).fit(fi, y)
    raw, _, _ = m._raw(fi)
    m.calibrator = PlattCal(on_probability=strategy in ("F0", "F5", "F6")).fit(raw, y)
    return m


# ---------------------------------------------------------------- thresholds & policy
@dataclass
class DecisionPolicy:
    t_high: float
    t_low: float
    veto: dict[str, dict]                      # module -> {"p_fraud": tau, "min_conf": c}
    targets: dict
    achieved_on_fit: dict = field(default_factory=dict)

    def decide(self, fused: float, P: dict[str, float | None], C: dict[str, float | None],
               challenge_failed: bool = False) -> tuple[str, str | None, str]:
        for mod, v in self.veto.items():
            p, c = P.get(mod), C.get(mod)
            if p is not None and c is not None and (1 - p) >= v["p_fraud"] and c >= v["min_conf"]:
                return "REJECT", mod, f"{mod} is highly confident of fraud (veto)."
        if challenge_failed:
            return "REJECT", "M4", "The liveness challenge was not completed. Please try again and follow the prompts."
        if fused >= self.t_high:
            return "ACCEPT", None, "All checks passed."
        if fused >= self.t_low:
            return "HUMAN_REVIEW", None, "Some checks were inconclusive; a reviewer will look at this session."
        return "REJECT", None, "The verification failed."


def choose_thresholds(p: np.ndarray, y: np.ndarray, attack_type: np.ndarray, apcer_max: float, bpcer_max: float):
    """T_high: smallest t with max per-attack-type acceptance rate <= apcer_max (worst-case APCER).
    T_low: largest t with genuine rejection rate (p < t) <= bpcer_max. If T_low > T_high -> T_low = T_high."""
    att = y == 0
    cand = np.unique(np.r_[p, 0.0, 1.0])
    t_high = 1.0
    for t in cand:
        worst = max(((p[att & (attack_type == a)] >= t).mean() for a in np.unique(attack_type[att])), default=0.0)
        if worst <= apcer_max:
            t_high = float(t)
            break
    g = np.sort(p[y == 1])
    k = int(math.floor(bpcer_max * len(g)))
    t_low = float(g[k]) if len(g) else 0.0          # exactly k genuine scores fall below t_low
    t_low = min(t_low, t_high)
    return t_high, t_low


def choose_veto(P: np.ndarray, C: np.ndarray, y: np.ndarray, modules: list[str], false_trigger_max: float,
                min_conf: float = 0.5) -> dict:
    """Per-module tau so that genuine sessions trigger the veto at most `false_trigger_max` of the time."""
    out = {}
    for mod in modules:
        i = MODULES.index(mod)
        g = (y == 1) & np.isfinite(P[:, i]) & (np.nan_to_num(C[:, i]) >= min_conf)
        if g.sum() < 50:
            continue
        pf = 1 - P[g, i]
        tau = float(np.quantile(pf, 1 - false_trigger_max)) + 1e-6
        out[mod] = {"p_fraud": min(max(tau, 0.9), 0.9999), "min_conf": min_conf, "n_genuine_fit": int(g.sum())}
    return out


def evaluate_policy(pol: DecisionPolicy, fused, P, C, y, attack_type) -> dict:
    dec = []
    for i in range(len(y)):
        Pi = {m: (None if not np.isfinite(P[i, j]) else float(P[i, j])) for j, m in enumerate(MODULES)}
        Ci = {m: (None if not np.isfinite(C[i, j]) else float(C[i, j])) for j, m in enumerate(MODULES)}
        dec.append(pol.decide(float(fused[i]), Pi, Ci)[0])
    dec = np.array(dec)
    g, a = y == 1, y == 0
    per_type = {t: {"n": int((attack_type == t).sum()), "accepted": float((dec[attack_type == t] == "ACCEPT").mean()),
                    "rejected": float((dec[attack_type == t] == "REJECT").mean())} for t in np.unique(attack_type[a])}
    return {"n_genuine": int(g.sum()), "n_attack": int(a.sum()),
            "genuine": {d: float((dec[g] == d).mean()) for d in ("ACCEPT", "HUMAN_REVIEW", "REJECT")},
            "attack": {d: float((dec[a] == d).mean()) for d in ("ACCEPT", "HUMAN_REVIEW", "REJECT")},
            "apcer_accept_overall": float((dec[a] == "ACCEPT").mean()),
            "apcer_accept_worst_type": max(v["accepted"] for v in per_type.values()) if per_type else None,
            "bpcer_reject": float((dec[g] == "REJECT").mean()),
            "review_rate": float((dec == "HUMAN_REVIEW").mean()),
            "per_attack_type": per_type}
