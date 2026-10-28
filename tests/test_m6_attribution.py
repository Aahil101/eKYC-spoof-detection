"""Tests for the forgery-attribution embedding (ml/attribution.py) and its use as a novelty space."""
import numpy as np

from ml.attribution import fit_embedder, ood_scores
from monitoring.novelty import fit_novelty


def _data(seed=0, n=120, d=40):
    rng = np.random.default_rng(seed)
    centers = rng.normal(0, 3, (5, d))
    X = np.vstack([rng.normal(c, 1, (n, d)) for c in centers])
    y = np.repeat([f"c{i}" for i in range(5)], n)
    return X.astype(np.float32), y


def test_embedder_learns_classes_and_flags_heldout():
    X, y = _data()
    known = y != "c4"
    emb = fit_embedder(X[known], y[known], seed=0, epochs=15, dim=16)
    E = emb.embed(X[:5])
    assert E.shape == (5, 16) and np.allclose(np.linalg.norm(E, axis=1), 1, atol=1e-4)
    acc = (np.array(emb.classes)[emb.logits(X[known]).argmax(1)] == y[known]).mean()
    assert acc > 0.9
    s_known = ood_scores(emb, X[known], y[known], X[known][::4])
    s_novel = ood_scores(emb, X[known], y[known], X[~known])
    from sklearn.metrics import roc_auc_score
    for k in ("knn", "maha"):
        auc = roc_auc_score(np.r_[np.zeros(len(s_known[k])), np.ones(len(s_novel[k]))], np.r_[s_known[k], s_novel[k]])
        assert auc > 0.8, (k, auc)


def test_supcon_novelty_space_with_pretrained_embedder_and_pickle():
    import pickle
    X, y = _data(1)
    emb = fit_embedder(X, y, seed=0, epochs=5, dim=8)
    nov = fit_novelty(X, list(y), seed=0, k=8, kind="supcon", use_sklearn=False, embedder=emb)
    assert nov.space.embedder is emb and nov.meta["space"] == "supcon"
    s = nov.scores(X[:10])
    assert s["mahalanobis"].shape == (10,)
    nov2 = pickle.loads(pickle.dumps(nov))           # served reference must pickle without torch objects
    assert np.allclose(nov2.scores(X[:10])["knn"], s["knn"])


def test_detect_actions_from_synthetic_head_pose():
    """Self-paced liveness: a synthetic landmark track that turns left/right, nods up/down and blinks."""
    from ml.common import face as F
    from ml.features.liveness import ACTIONS, detect_actions

    def frame(t, dx=0.0, dy=0.0, eye_open=1.0):
        L = np.zeros((478, 3))
        L[list(F.LEFT_EYE_EAR), 0], L[list(F.RIGHT_EYE_EAR), 0] = 140, 60      # eye centres x
        L[list(F.LEFT_EYE_EAR), 1] = L[list(F.RIGHT_EYE_EAR), 1] = 100
        # eye opening: vertical spread of the EAR points (indices 1,2 above / 4,5 below)
        for eye in (F.LEFT_EYE_EAR, F.RIGHT_EYE_EAR):
            x0 = L[eye[0], 0]
            L[eye[0], 0], L[eye[3], 0] = x0 - 15, x0 + 15
            for j in (1, 2):
                L[eye[j], 1] = 100 - 6 * eye_open
            for j in (4, 5):
                L[eye[j], 1] = 100 + 6 * eye_open
        L[F.NOSE_TIP, :2] = (100 + dx * 80, 150 + dy * 100)
        L[F.CHIN, :2] = (100, 200)
        return F.FaceFrame(t_ms=t, landmarks=L, bbox=(40, 60, 160, 210))
    seq = [(0, 0, 1)] * 30 + [(0.4, 0, 1)] * 5 + [(0, 0, 1)] * 10 + [(-0.4, 0, 1)] * 5 + [(0, 0, 1)] * 10 \
        + [(0, -0.2, 1)] * 5 + [(0, 0, 1)] * 10 + [(0, 0.2, 1)] * 5 + [(0, 0, 1)] * 5 + [(0, 0, 0.05)] * 2 + [(0, 0, 1)] * 10
    faces = [frame(i * 125.0, dx, dy, eo) for i, (dx, dy, eo) in enumerate(seq)]
    out = detect_actions(faces, [f.t_ms for f in faces])
    got = {a["action"]: a["done"] for a in out["actions"]}
    assert set(got) == set(ACTIONS)
    assert all(got.values()) and out["all_done"], out["actions"]
    still = [frame(i * 125.0) for i in range(60)]
    out2 = detect_actions(still, [f.t_ms for f in still])
    assert not out2["all_done"] and not any(a["done"] for a in out2["actions"] if a["action"] != "blink")
