"""Milestone 4 unit tests: fusion strategies, DS combination, thresholds, veto, policy."""
import numpy as np
import pytest

from ml.fusion import (MODULES, DecisionPolicy, FusionInput, choose_thresholds, choose_veto, ds_combine,
                       evaluate_policy, fit_fusion, quality_matrix)


def _synthetic(n=2000, seed=0, blur_breaks_m2=True):
    """5 modules; module 2 is unreliable when the 'blur' quality feature is low (gating should learn this)."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    blur = rng.uniform(0, 1, n)
    P = np.empty((n, 5))
    C = np.empty((n, 5))
    for i in range(5):
        strength = np.full(n, 0.35)
        if i == 1 and blur_breaks_m2:
            strength = np.where(blur < 0.5, 0.0, 3.0)       # pure noise when blurry, strong when sharp
        z = (2 * y - 1) * strength + rng.normal(0, 1, n)
        P[:, i] = 1 / (1 + np.exp(-z))
        C[:, i] = np.clip(np.abs(P[:, i] - 0.5) * 2, 0, 1)
    P[rng.random(n) < 0.1, 4] = np.nan                       # M5 missing sometimes
    C[np.isnan(P)] = np.nan
    Q = quality_matrix([{"blur_laplacian_var": float(np.expm1(b * 6))} for b in blur])
    return FusionInput(P, C, Q), y


def test_ds_combine_properties():
    P = np.array([[0.9, 0.9], [0.9, 0.1], [0.9, np.nan]])
    C = np.array([[0.8, 0.8], [0.8, 0.8], [0.8, np.nan]])
    p, u = ds_combine(P, C)
    assert p[0] > 0.9 and u[0] < 0.2 * 0.2 + 1e-9 * 0 + 0.05     # agreement strengthens, uncertainty shrinks
    assert abs(p[1] - 0.5) < 1e-6                                  # symmetric conflict -> undecided
    p_single, _ = ds_combine(P[2:, :1], C[2:, :1])
    assert p[2] == pytest.approx(p_single[0])                      # missing module = vacuous opinion
    p0, u0 = ds_combine(np.array([[0.99]]), np.array([[0.0]]))
    assert p0[0] == pytest.approx(0.5) and u0[0] == pytest.approx(1.0)   # zero confidence carries no evidence


@pytest.mark.parametrize("st", ["F0", "F1", "F2", "F3", "F4", "F5", "F6"])
def test_strategies_beat_chance_and_handle_missing(st):
    from sklearn.metrics import roc_auc_score
    tr, ytr = _synthetic(seed=0)
    te, yte = _synthetic(seed=1)
    m = fit_fusion(st, tr, ytr, seed=0)
    p, w, u = m.predict(te)
    assert np.all((p > 0) & (p < 1)) and p.shape == (len(yte),)
    assert roc_auc_score(yte, p) > 0.75
    if st in ("F0", "F1", "F4"):
        miss = np.isnan(te.P[:, 4])
        assert np.allclose(w[miss, 4], 0)                            # missing module gets no weight
        assert np.allclose(w.sum(1), 1)


def test_quality_gate_learns_to_downweight_unreliable_module():
    tr, ytr = _synthetic(4000, seed=0)
    m = fit_fusion("F4", tr, ytr, seed=0)
    te, _ = _synthetic(4000, seed=2)
    w = m.gate.weights(te)
    blur = te.Q[:, 0]
    sharp, blurry = blur > np.nanmedian(blur), blur <= np.nanmedian(blur)
    assert w[sharp, 1].mean() > 2 * w[blurry, 1].mean()             # M2 trusted only on sharp input


def test_gated_fusion_beats_fixed_on_quality_dependent_module():
    from sklearn.metrics import roc_auc_score
    tr, ytr = _synthetic(4000, seed=0)
    te, yte = _synthetic(4000, seed=3)
    a1 = roc_auc_score(yte, fit_fusion("F1", tr, ytr).predict(te)[0])
    a4 = roc_auc_score(yte, fit_fusion("F4", tr, ytr).predict(te)[0])
    assert a4 > a1 + 0.01


def test_thresholds_meet_targets():
    rng = np.random.default_rng(0)
    y = np.r_[np.ones(2000), np.zeros(2000)].astype(int)
    at = np.array(["bona_fide"] * 2000 + ["a"] * 1000 + ["b"] * 1000)
    p = np.r_[rng.beta(5, 2, 2000), rng.beta(2, 5, 1000), rng.beta(2, 3, 1000)]   # attack b is harder
    th, tl = choose_thresholds(p, y, at, apcer_max=0.01, bpcer_max=0.02)
    assert tl <= th
    assert (p[at == "b"] >= th).mean() <= 0.01 and (p[at == "a"] >= th).mean() <= 0.01   # worst type
    assert (p[y == 1] < tl).mean() <= 0.02
    assert (p[y == 1] < tl).mean() >= 0.015                       # and not needlessly conservative


def test_veto_and_policy():
    rng = np.random.default_rng(0)
    n = 3000
    y = np.r_[np.ones(n), np.zeros(n)].astype(int)
    P = np.full((2 * n, 5), 0.9)
    C = np.full((2 * n, 5), 0.9)
    P[:n, 1] = rng.beta(8, 1, n)                                    # genuine: M2 high
    P[n:, 1] = rng.beta(1, 40, n)                                   # attacks: M2 very low (veto floor is 0.9)
    assert choose_veto(P[:1000], C[:1000], y[:1000], ["M2"], 0.001)["M2"]["disabled"]   # too few genuine
    veto = choose_veto(P, C, y, ["M2"], false_trigger_max=0.001, min_conf=0.5)
    tau = veto["M2"]["p_fraud"]
    assert ((1 - P[:n, 1]) >= tau).mean() <= 0.002
    pol = DecisionPolicy(0.95, 0.3, veto, {})
    Pd = {m: 0.99 for m in MODULES}
    Cd = {m: 0.99 for m in MODULES}
    assert pol.decide(0.99, Pd, Cd)[0] == "ACCEPT"
    assert pol.decide(0.5, Pd, Cd)[0] == "HUMAN_REVIEW"
    assert pol.decide(0.1, Pd, Cd)[0] == "REJECT"
    Pd["M2"] = 1 - tau + 0.0001                                     # fraud prob just below the veto threshold
    assert pol.decide(0.99, Pd, Cd)[0] == "ACCEPT"
    Pd["M2"] = 0.0
    d, mod, _ = pol.decide(0.99, Pd, Cd)
    assert d == "REJECT" and mod == "M2"                            # veto overrides a high fused score
    Cd["M2"] = 0.1
    assert pol.decide(0.99, Pd, Cd)[0] == "ACCEPT"                  # low-confidence module cannot veto
    assert pol.decide(0.99, {m: 0.99 for m in MODULES}, {m: 0.99 for m in MODULES}, challenge_failed=True)[0] == "REJECT"
    e = evaluate_policy(pol, np.r_[np.full(n, 0.99), np.full(n, 0.99)], P, C, y, np.array(["bona_fide"] * n + ["x"] * n))
    assert e["per_attack_type"]["x"]["rejected"] > 0.9               # attacks caught by veto alone
