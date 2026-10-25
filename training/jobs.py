"""Training job registry.

Each job is `fn(cfg, ctx) -> dict(metrics)`. Jobs MUST:
  * be deterministic given cfg["seed"];
  * call ctx.maybe_checkpoint(step, state) regularly and restore from ctx.resume_state;
  * write final artifacts with ctx.save_artifact(name, bytes_or_path);
  * stop cleanly when ctx.out_of_time() is True (Kaggle 12 h limit).

Milestone 2 ships `selftest_logreg`: a small numpy logistic regression on synthetic
data. It exists ONLY to verify the checkpoint/resume/notebook/import plumbing on
both the laptop and Kaggle. Its "metrics" describe synthetic data and are not
research results. Module training jobs (M1-M5) are added in milestone 3; their
configs exist already with `implemented: false`.
"""
from __future__ import annotations

import io
import json
from typing import Any, Callable

import numpy as np

JobFn = Callable[[dict, Any], dict]
JOBS: dict[str, JobFn] = {}


def register(name: str):
    def deco(fn: JobFn) -> JobFn:
        JOBS[name] = fn
        return fn
    return deco


@register("selftest_logreg")
def selftest_logreg(cfg: dict, ctx) -> dict:
    p = cfg["params"]
    rng = np.random.default_rng(cfg["seed"])
    n, d = int(p["n_samples"]), int(p["n_features"])
    w_true = rng.normal(size=d)
    X = rng.normal(size=(n, d))
    y = (X @ w_true + rng.normal(scale=0.5, size=n) > 0).astype(np.float64)

    state = ctx.resume_state or {"step": 0, "w": np.zeros(d), "b": 0.0,
                                 "rng": np.random.default_rng(cfg["seed"] + 1).bit_generator.state}
    w, b, step = state["w"], state["b"], state["step"]
    batch_rng = np.random.default_rng()
    batch_rng.bit_generator.state = state["rng"]
    total, lr, bs = int(p["steps"]), float(p["lr"]), int(p["batch_size"])
    fail_at = p.get("simulate_crash_at_step")

    while step < total:
        if ctx.out_of_time():
            ctx.log("budget_exhausted", step=step)
            break
        idx = batch_rng.integers(0, n, size=bs)
        z = X[idx] @ w + b
        g = 1.0 / (1.0 + np.exp(-z)) - y[idx]
        w = w - lr * (X[idx].T @ g) / bs
        b = b - lr * g.mean()
        step += 1
        ctx.maybe_checkpoint(step, {"step": step, "w": w, "b": b, "rng": batch_rng.bit_generator.state})
        if fail_at is not None and step == int(fail_at):
            raise RuntimeError(f"simulated crash at step {step}")

    prob = 1.0 / (1.0 + np.exp(-(X @ w + b)))
    acc = float(((prob > 0.5) == y).mean())
    buf = io.BytesIO()
    np.savez(buf, w=w, b=np.array([b]))
    ctx.save_artifact("head.npz", buf.getvalue())
    ctx.save_artifact("calibrator.json", json.dumps({"type": "identity", "note": "selftest"}).encode())
    return {"completed_steps": step, "finished": step >= total, "train_accuracy_synthetic": acc,
            "note": "synthetic plumbing test - not a research result"}
