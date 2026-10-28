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
