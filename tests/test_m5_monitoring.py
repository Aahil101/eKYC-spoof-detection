"""Milestone 5: drift tests, streaming detectors, novelty, disambiguation on synthetic data with KNOWN shifts."""
import numpy as np
import pytest

from monitoring import drift as D
from monitoring.disambiguation import analyse_window, build_null
from monitoring.novelty import binomial_excess, cluster_candidates, fit_novelty
from monitoring.recommend import recommend, sensitivity_table


def test_ks_psi_bh_detect_shift_and_control_false_alarms():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(1000, 6))
    cur = rng.normal(size=(300, 6))
    cur[:, 2] -= 1.0                                     # only feature 2 shifts
    rows = D.per_feature_tests(ref, cur, [f"f{i}" for i in range(6)])
    assert [r["feature"] for r in rows if r["drifted"]] == ["f2"]
    assert rows[2]["psi"] > rows[2]["psi_crit"] and rows[2]["mean_shift_sd"] < -0.8
    # false-alarm rate under H0 across repeated windows stays near/below alpha (BH controls FDR)
    fa = 0
    for s in range(40):
        r0 = D.per_feature_tests(ref, np.random.default_rng(s + 1).normal(size=(300, 6)), [f"f{i}" for i in range(6)])
        fa += any(x["drifted"] for x in r0)
    assert fa / 40 <= 0.15


def test_psi_critical_value_matches_null():
    rng = np.random.default_rng(0)
    null = [D.psi(rng.normal(size=1000), rng.normal(size=300)) for _ in range(300)]
    crit = D.psi_critical(1000, 300)
    assert 0.02 <= np.mean(np.array(null) > crit) <= 0.10


def test_bh_monotone_and_known_values():
    q = D.benjamini_hochberg(np.array([0.01, 0.04, 0.03, 0.5, np.nan]))
    assert np.allclose(q[:4], [0.04, 0.04 * 0 + 0.0533333, 0.0533333, 0.5], atol=1e-4)
    assert np.isnan(q[4])


def test_mmd_and_c2st():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(200, 5)), rng.normal(size=(200, 5))
    assert D.mmd_test(a, b, n_perm=100)["p_value"] > 0.05
    b2 = b.copy()
    b2[:, 0] += 0.8
    assert D.mmd_test(a, b2, n_perm=100)["p_value"] < 0.05
    c = D.c2st(a, b2, [f"f{i}" for i in range(5)])
    assert c["p_value"] < 0.05 and c["top_features"][0]["feature"] == "f0"


def test_density_ratio_weighting_removes_explained_shift():
    """Artifact A depends on quality q; shifting q shifts A. Reweighting the reference by w(q) explains it."""
    rng = np.random.default_rng(0)
    q_ref = rng.normal(size=(1500, 1))
    a_ref = q_ref * 1.5 + rng.normal(0, 0.5, (1500, 1))
    q_cur = rng.normal(1.0, 1, (300, 1))
    a_cur = q_cur * 1.5 + rng.normal(0, 0.5, (300, 1))
    assert D.mmd_test(a_ref, a_cur, 100)["p_value"] < 0.05
    w = D.density_ratio_weights(q_ref, q_cur)
    assert D.mmd_test(a_ref, a_cur, 100, weights_ref=w)["p_value"] > 0.05


@pytest.mark.parametrize("det", ["adwin", "ph"])
def test_streaming_mean_change_detected_with_small_delay(det):
    rng = np.random.default_rng(0)
    x = np.r_[rng.normal(0, 1, 600), rng.normal(2.0, 1, 400)]
    d = D.ADWIN(delta=0.002) if det == "adwin" else D.PageHinkley(delta=0.05, threshold=30)
    hits = [i for i, v in enumerate(x) if d.update(v)]
    pre = [h for h in hits if h < 600]
    post = [h for h in hits if h >= 600]
    assert len(pre) <= 1 and post and post[0] - 600 < 120


def test_ddm_eddm_error_rate_increase():
    rng = np.random.default_rng(0)
    e = np.r_[rng.random(1500) < 0.05, rng.random(1500) < 0.30].astype(int)
    for det in (D.DDM(), D.EDDM()):
        hits = [i for i, v in enumerate(e) if det.update(int(v))]
        assert any(h >= 1500 for h in hits) and all(h > 300 for h in hits)


def _world(n, seed, quality_shift=0.0, novel_frac=0.0, novel_center=6.0):
    """Two known classes in a 10-d artifact space; quality q (3-d) shifts artifacts slightly (explainable)."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    q = rng.normal(quality_shift, 1, (n, 3))
    A = rng.normal(0, 1, (n, 10))
    A[:, 0] += 3 * y
    A[:, 1] += 0.3 * q[:, 0]
    k = int(novel_frac * n)
    if k:
        A[:k] = rng.normal(0, 0.3, (k, 10))
        A[:k, 2] += novel_center                          # compact new cluster away from both classes
        y[:k] = 0
    labels = np.where(y == 1, "bona_fide", "deepfake_x").astype(object)
    if k:
        labels[:k] = "novel"
    fused = 1 / (1 + np.exp(-(3 * y - 1.5 + rng.normal(0, 1, n))))
    dec = np.where(fused > 0.7, "ACCEPT", np.where(fused > 0.3, "HUMAN_REVIEW", "REJECT"))
    return {"Q": q, "A": A, "fused": fused, "decisions": dec}, labels


@pytest.fixture(scope="module")
def monitor_setup():
    ref, lab = _world(1500, 0)
    nov = fit_novelty(ref["A"], list(lab), seed=0, k=8, use_sklearn=False)
    null = build_null(ref, [_world(200, 100 + i)[0] for i in range(40)], nov, ["q0", "q1", "q2"],
                      {"n_perm": 50}, seed=0)
    return ref, nov, null


@pytest.mark.parametrize("scenario,expected", [
    ("stationary", {"NONE"}),
    ("quality", {"QUALITY_DRIFT"}),
    ("novel", {"NOVEL_ATTACK"}),
    ("mixed", {"MIXED"}),
])
def test_disambiguation_on_known_scenarios(monitor_setup, scenario, expected):
    ref, nov, null = monitor_setup
    kw = {"stationary": {}, "quality": {"quality_shift": 1.0}, "novel": {"novel_frac": 0.15},
          "mixed": {"quality_shift": 1.0, "novel_frac": 0.15}}[scenario]
    cur, _ = _world(200, 7, **kw)
    rep = analyse_window(ref, cur, nov, ["q0", "q1", "q2"], {"n_perm": 50}, seed=1, null=null)
    assert rep["label"] in expected, (rep["label"], rep["tests"])
    if scenario in ("novel", "mixed"):
        assert rep["clusters"]["coherent"] and rep["clusters"]["clusters"][0]["size"] >= 15


def test_binomial_excess_and_clusters():
    rng = np.random.default_rng(0)
    assert binomial_excess(rng.uniform(size=500))["p_value"] > 0.05
    pv = np.r_[rng.uniform(size=400), np.full(100, 0.001)]
    assert binomial_excess(pv)["p_value"] < 1e-6
    Z = np.r_[rng.normal(0, 0.2, (40, 3)), rng.uniform(-5, 5, (10, 3))]
    c = cluster_candidates(Z, 8, eps=0.3)
    assert max(x["size"] for x in c["clusters"]) >= 35
    assert len(c["clusters"]) == 1                                     # scattered points stay noise


def test_recommendation_names_feature_and_module():
    rng = np.random.default_rng(0)
    n = 900
    y = rng.integers(0, 2, n)
    Q = rng.normal(size=(n, 2))
    P = np.full((n, 5), np.nan)
    good = Q[:, 0] > -0.4                                   # M2 only works on sharp frames
    P[:, 1] = np.where(good, np.clip(y * 0.8 + rng.normal(0.1, 0.1, n), 0, 1), rng.uniform(size=n))
    sens = sensitivity_table(Q, ["blur_laplacian_var", "brightness_mean"], P, y)
    rep = {"label": "QUALITY_DRIFT", "params": {"alpha_novel": 0.05},
           "top_drifting_features": [{"feature": "blur_laplacian_var", "drifted": True, "psi": 0.41, "psi_crit": 0.05,
                                      "ks_q": 1e-6, "mean_shift_sd": -1.2}]}
    r = recommend(rep, sens, "14:00-15:00", "v0.4")
    assert r and r[0]["module"] == "M2" and "blur" in r[0]["message"] and "14:00-15:00" in r[0]["message"]



def _pool(n, seed, source, cls_shift=0.0, q_shift=0.0, novel=False):
    """Stream-pool shaped records (forgery-specific artifact vectors) for the service test."""
    from monitoring.novelty import ARTIFACT_DIM
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n) if not novel else np.zeros(n, int)
    A = rng.normal(0, 1, (n, ARTIFACT_DIM)).astype(np.float32)
    A[:, 0] += 3 * y
    if novel:
        A = rng.normal(0, 0.3, (n, ARTIFACT_DIM)).astype(np.float32)
        A[:, 5] += 8
    from ml.fusion import QUALITY_KEYS
    Q = rng.normal(q_shift, 1, (n, len(QUALITY_KEYS)))
    fused = 1 / (1 + np.exp(-(3 * y - 1.5 + rng.normal(0, 1, n))))
    return {"P": np.tile(fused[:, None], 5), "C": np.full((n, 5), 0.8), "Q": Q, "A": A, "fused": fused,
            "decisions": np.where(fused > 0.5, "ACCEPT", "REJECT"), "y": y,
            "attack_type": np.where(y == 1, "bona_fide", "deepfake_x"), "video_class": np.where(y == 1, "bona_fide", "deepfake_x"),
            "source": np.full(n, source), "video_id": np.array([f"{source}{seed}_{i}" for i in range(n)]),
            "is_burn_in": np.zeros(n, bool)}


def test_service_simulate_window_report_alert(tmp_path):
    from backend.app.db import Alert, DriftReport, init_db, make_engine, make_sessionmaker
    from ml.fusion import QUALITY_KEYS
    from monitoring.service import run_window, simulate
    ref_d = _pool(1200, 0, "ffpp")
    names = list(QUALITY_KEYS)
    nov = fit_novelty(ref_d["A"], list(ref_d["video_class"]), seed=0, k=8, use_sklearn=False)
    null = build_null(ref_d, [_pool(200, 50 + i, "ffpp") for i in range(30)], nov, names, {"n_perm": 30}, 0)
    pool = {k: np.concatenate([a[k], b[k]]) for a, b in [(_pool(400, 1, "ffpp"), _pool(300, 2, "celebdf", q_shift=1.5))]
            for k in a}
    ref = {"quality_names": names, "reference": ref_d, "novelty": nov, "null": null,
           "sensitivity": sensitivity_table(ref_d["Q"], names, ref_d["P"], ref_d["y"]), "window": 200,
           "monitor": {"n_perm": 30}, "stream_pool": pool}
    eng = make_engine(f"sqlite:///{(tmp_path / 'm.db').as_posix()}")
    init_db(eng)
    sm = make_sessionmaker(eng)
    assert run_window(sm, ref, "v1")["status"] == "insufficient_data"
    assert simulate(sm, ref, "v1", "stationary", 200, 0)["inserted"] > 150
    r0 = run_window(sm, ref, "v1", 200, source="simulated")
    assert r0["status"] == "ok" and r0["label"] == "NONE"
    simulate(sm, ref, "v1", "domain_drift", 200, 1)
    r1 = run_window(sm, ref, "v1", 200, source="simulated")
    assert r1["label"] in ("QUALITY_DRIFT", "CONCEPT_DRIFT", "MIXED")
    with sm() as db:
        assert db.query(DriftReport).count() == 2
        al = db.query(Alert).all()
        assert len(al) == 1 and al[0].bundle_version == "v1" and al[0].evidence["tests"]["H_Q"]
