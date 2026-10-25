"""Config-driven training runner (laptop smoke mode and Kaggle full mode).

    python -m training.run --config training/configs/selftest.yaml --mode smoke
    python -m training.run --config training/configs/m2_clip_probe.yaml --mode full --resume auto

Output layout (Kaggle: /kaggle/working/runs/<run_name>/, laptop: ./runs/<run_name>/):
    checkpoints/ckpt_*.pkl|pt, checkpoints/latest.json
    artifacts/<files for the model bundle>
    run_manifest.json   (config, git sha, env, data manifest, artifact hashes, metrics, status)
    metrics.json
    log.jsonl
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import platform
import random
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np

from training import jobs as jobs_mod
from training import audit as _audit  # noqa: F401  (registers dataset_audit)
from training import fetch_job as _fetch  # noqa: F401  (registers fetch_archive)
from training.checkpoint import CheckpointManager, find_resume_checkpoint
from training.config import load_job_config
from training.paths import REPO_ROOT, input_roots, is_kaggle, output_root


def _git_sha() -> str:
    if sha := os.environ.get("CODMAV_GIT_SHA"):
        return sha
    stamp = REPO_ROOT / "GIT_SHA"
    if stamp.exists():
        return stamp.read_text().strip()
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL,
                                       text=True).strip()
    except Exception:
        return "unknown"


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


class RunContext:
    def __init__(self, cfg: dict, run_dir: Path, resume_state: dict | None):
        self.cfg = cfg
        self.run_dir = run_dir
        self.art_dir = run_dir / "artifacts"
        self.art_dir.mkdir(parents=True, exist_ok=True)
        ck = cfg.get("checkpoint", {})
        self.ckpt = CheckpointManager(run_dir / "checkpoints", ck.get("every_steps", 500),
                                      ck.get("every_minutes", 10), ck.get("keep_last", 3))
        self.resume_state = resume_state
        self._t0 = time.monotonic()
        self._budget_s = float(cfg.get("budget", {}).get("max_wall_minutes", 0)) * 60.0
        self._log = open(run_dir / "log.jsonl", "a", encoding="utf-8")

    def out_of_time(self) -> bool:
        return self._budget_s > 0 and time.monotonic() - self._t0 >= self._budget_s

    def maybe_checkpoint(self, step: int, state: dict, force: bool = False) -> None:
        if force or self.ckpt.should_save(step):
            p = self.ckpt.save(state, step)
            self.log("checkpoint", step=step, file=p.name)

    def save_artifact(self, name: str, data: bytes | Path) -> Path:
        target = self.art_dir / name
        tmp = target.with_suffix(target.suffix + ".tmp")
        tmp.write_bytes(data if isinstance(data, bytes) else Path(data).read_bytes())
        os.replace(tmp, target)
        return target

    def log(self, event: str, **kw) -> None:
        rec = {"t": dt.datetime.now(dt.timezone.utc).isoformat(), "event": event, **kw}
        self._log.write(json.dumps(rec, default=str) + "\n")
        self._log.flush()
        print(json.dumps(rec, default=str), flush=True)

    def close(self) -> None:
        self._log.close()


def run(config_path: Path, mode: str, resume: str = "auto", run_name: str | None = None,
        out_root: Path | None = None) -> dict:
    cfg = load_job_config(config_path, mode)
    if run_name:
        cfg["run_name"] = run_name
    if not cfg.get("implemented", True):
        raise SystemExit(f"job {cfg['job']!r} is planned but not implemented yet (milestone 3). "
                         f"Config: {config_path}")
    if cfg["job"] not in jobs_mod.JOBS:
        raise SystemExit(f"unknown job {cfg['job']!r}; known: {sorted(jobs_mod.JOBS)}")
    run_dir = (out_root or output_root()) / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)

    resume_path = None
    if resume == "auto":
        resume_path = find_resume_checkpoint(cfg["run_name"], run_dir / "checkpoints", input_roots())
    elif resume not in ("none", ""):
        resume_path = Path(resume)
    resume_state = CheckpointManager.load(resume_path) if resume_path else None

    seed_everything(int(cfg["seed"]))
    ctx = RunContext(cfg, run_dir, resume_state)
    ctx.log("start", job=cfg["job"], mode=mode, kaggle=is_kaggle(),
            resumed_from=str(resume_path) if resume_path else None)

    manifest = {
        "run_name": cfg["run_name"], "job": cfg["job"], "mode": mode, "config": cfg,
        "config_sha256": cfg["config_sha256"], "git_sha": _git_sha(),
        "started_at": dt.datetime.now(dt.timezone.utc).isoformat(),
        "env": {"python": sys.version.split()[0], "platform": platform.platform(), "kaggle": is_kaggle(),
                "kaggle_kernel": os.environ.get("KAGGLE_KERNEL_RUN_TYPE")},
        "resumed_from": str(resume_path) if resume_path else None,
        "data": cfg.get("data", {}),
    }
    status = "failed"
    metrics: dict = {}
    try:
        metrics = jobs_mod.JOBS[cfg["job"]](cfg, ctx)
        status = "completed" if metrics.get("finished", True) else "incomplete_budget"
    except Exception as e:
        ctx.log("error", error=f"{type(e).__name__}: {e}", tb=traceback.format_exc())
        raise
    finally:
        manifest["status"] = status
        manifest["finished_at"] = dt.datetime.now(dt.timezone.utc).isoformat()
        manifest["metrics"] = metrics
        manifest["artifacts"] = {p.name: _sha256(p) for p in sorted(ctx.art_dir.iterdir())
                                 if p.is_file() and not p.name.endswith(".tmp")}
        (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
        (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
        ctx.log("end", status=status)
        ctx.close()
    return manifest


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--mode", choices=["smoke", "full"], default="smoke")
    ap.add_argument("--resume", default="auto", help="auto | none | <checkpoint path>")
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--out-root", type=Path, default=None)
    a = ap.parse_args(argv)
    m = run(a.config, a.mode, a.resume, a.run_name, a.out_root)
    print(f"[codmav] run {m['run_name']} status={m['status']} artifacts={list(m['artifacts'])}")


if __name__ == "__main__":
    main()
