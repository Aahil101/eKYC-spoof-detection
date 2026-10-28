"""Forgery-attribution embedding for novel-attack detection (E6/E7 follow-up).

Motivation (docs/EXPERIMENTS.md E6): in unsupervised spaces (PCA/LDA of CLIP + spectrum) a held-out manipulation
method is not out-of-distribution, because the dominant variance is identity/scene, not forgery traces. Here a
small network is trained to ATTRIBUTE known classes (bona fide + each known method / replay instrument) with
cross-entropy + supervised contrastive loss (Khosla et al., NeurIPS 2020). The L2-normalised embedding is then
used for OOD scoring with k-NN distance (Sun et al., ICML 2022) or class-conditional Mahalanobis (Lee et al. 2018).

Training uses torch (CPU is enough: a few thousand vectors); the fitted model is stored as numpy weights so the
serving container needs no torch to embed. Inputs are standardised with training statistics.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DEFAULTS = {"hidden": 256, "dim": 64, "epochs": 60, "lr": 1e-3, "weight_decay": 1e-4, "batch": 256,
            "temperature": 0.1, "supcon_weight": 1.0, "dropout": 0.2, "input_noise": 0.1}


@dataclass
class AttributionEmbedder:
    mu: np.ndarray
    sd: np.ndarray
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray
    Wc: np.ndarray            # classifier head on the normalised embedding (for MSP / energy scores)
    bc: np.ndarray
    classes: list[str]
    meta: dict = field(default_factory=dict)

    def embed(self, X: np.ndarray) -> np.ndarray:
        Z = np.nan_to_num((np.asarray(X, np.float32) - self.mu) / self.sd)
        H = np.maximum(Z @ self.W1.T + self.b1, 0)
        E = H @ self.W2.T + self.b2
        return E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-9)

    def logits(self, X: np.ndarray) -> np.ndarray:
        return self.embed(X) @ self.Wc.T + self.bc


def _supcon(z, y, t):
    import torch
    sim = z @ z.T / t
    n = len(y)
    eye = torch.eye(n, dtype=torch.bool)
    sim = sim.masked_fill(eye, -1e9)
    pos = (y[:, None] == y[None]) & ~eye
    logp = sim - torch.logsumexp(sim, 1, keepdim=True)
    cnt = pos.sum(1)
    ok = cnt > 0
    return -((logp * pos).sum(1)[ok] / cnt[ok]).mean()


def fit_embedder(X: np.ndarray, labels, seed: int = 0, **kw) -> AttributionEmbedder:
    import torch
    p = {**DEFAULTS, **kw}
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels).astype(str)
    classes = sorted(set(labels))
    y = np.array([classes.index(l) for l in labels])
    mu = np.nanmean(X, 0)
    mu = np.where(np.isfinite(mu), mu, 0).astype(np.float32)
    sd = np.nanstd(X, 0)
    sd = np.where(np.isfinite(sd) & (sd > 1e-9), sd, 1).astype(np.float32)
    Z = torch.tensor(np.nan_to_num((X - mu) / sd), dtype=torch.float32)
    Y = torch.tensor(y)
    d_in = Z.shape[1]
    enc = torch.nn.Sequential(torch.nn.Dropout(p["dropout"]), torch.nn.Linear(d_in, p["hidden"]), torch.nn.ReLU(),
                              torch.nn.Linear(p["hidden"], p["dim"]))
    head = torch.nn.Linear(p["dim"], len(classes))
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=p["lr"], weight_decay=p["weight_decay"])
    # class-balanced sampling: every batch sees every class (SupCon needs positives)
    w = 1.0 / np.bincount(y)[y]
    w = w / w.sum()
    steps = max(1, len(y) // p["batch"])
    for ep in range(p["epochs"]):
        enc.train()
        for _ in range(steps):
            idx = torch.tensor(rng.choice(len(y), p["batch"], p=w))
            xb = Z[idx] + p["input_noise"] * torch.randn(len(idx), d_in)
            e = torch.nn.functional.normalize(enc(xb), dim=1)
            loss = torch.nn.functional.cross_entropy(head(e) * 10.0, Y[idx]) + p["supcon_weight"] * _supcon(e, Y[idx], p["temperature"])
            opt.zero_grad()
            loss.backward()
            opt.step()
    enc.eval()
    lin1, lin2 = enc[1], enc[3]
    g = lambda t: t.detach().numpy().astype(np.float32)
    return AttributionEmbedder(mu, sd, g(lin1.weight), g(lin1.bias), g(lin2.weight), g(lin2.bias),
                               g(head.weight) * 10.0, g(head.bias) * 10.0, classes,
                               {"params": p, "n": int(len(y)), "class_counts": {c: int((labels == c).sum()) for c in classes},
                                "final_loss": float(loss.detach())})


def ood_scores(emb: AttributionEmbedder, train_X: np.ndarray, train_labels, X: np.ndarray, k: int = 10,
               train_E: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """Higher = more novel. knn: k-th NN cosine distance to training embeddings; maha: min class-conditional
    Mahalanobis (shared covariance) in embedding space; msp: 1 - max softmax; energy: -logsumexp(logits)."""
    Et = emb.embed(train_X) if train_E is None else train_E
    E = emb.embed(X)
    out = []
    for i in range(0, len(E), 1024):
        d = 1 - E[i:i + 1024] @ Et.T
        out.append(np.partition(d, k - 1, axis=1)[:, k - 1])
    s = {"knn": np.concatenate(out) if out else np.zeros(0)}
    lab = np.asarray(train_labels).astype(str)
    cls = sorted(set(lab))
    means = np.stack([Et[lab == c].mean(0) for c in cls])
    R = np.vstack([Et[lab == c] - means[j] for j, c in enumerate(cls)])
    prec = np.linalg.inv(np.cov(R.T) + 1e-3 * np.eye(Et.shape[1]))
    dm = np.stack([np.einsum("ij,jk,ik->i", E - m, prec, E - m) for m in means], 1)
    s["maha"] = np.sqrt(np.maximum(dm.min(1), 0))
    L = E @ emb.Wc.T + emb.bc
    from scipy.special import logsumexp, softmax
    s["msp"] = 1 - softmax(L, 1).max(1)
    s["energy"] = -logsumexp(L, 1)
    return s
